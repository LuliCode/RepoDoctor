"""Java 8 AST chunks; token boundaries never count braces in comments/strings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import javalang
from javalang import tree as jt

from .models import Passage, digest
from .symbols import code_identifiers, is_entity_candidate

LOW_INFORMATION = set(
    "this super null true false void int long byte short boolean char float double String Object".split()
)
IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def identifiers(text: str) -> list[str]:
    """Compatibility alias for the shared, qualified symbol extractor."""
    return code_identifiers(text)


def type_name(node) -> str:
    if node is None:
        return "void"
    name = node.name
    child = getattr(node, "sub_type", None)
    while child:
        name += "." + child.name
        child = child.sub_type
    return name + "[]" * len(getattr(node, "dimensions", None) or [])


def ast_entities(node, field_names=()) -> set[str]:
    """Collect typed code anchors without treating local bindings as global nodes."""
    entities = set()

    def walk(value, parent=None):
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from walk(item, parent)
        elif isinstance(value, javalang.ast.Node):
            yield parent, value
            for attribute in value.attrs:
                yield from walk(getattr(value, attribute), value)

    nodes = list(walk(node))
    parents = {id(child): parent for parent, child in nodes}
    local_names = set()
    local_bindings = {}
    type_parameters = set()
    scopes = (
        jt.MethodDeclaration, jt.ConstructorDeclaration, jt.LambdaExpression,
        jt.BlockStatement, jt.ForStatement, jt.CatchClause, jt.TryStatement,
    )

    def enclosing_scope(child):
        current = parents.get(id(child))
        while current is not None:
            if isinstance(current, scopes):
                return current
            current = parents.get(id(current))
        return node

    def bind(name, child, *, scope=None, position=None):
        local_names.add(name)
        local_bindings.setdefault(name, []).append((
            enclosing_scope(child) if scope is None else scope, position,
        ))

    def locally_bound(name, child):
        """Only suppress a field name when a visible lexical binding can shadow it."""
        ancestors = {id(child), id(node)}
        current = parents.get(id(child))
        while current is not None:
            ancestors.add(id(current))
            current = parents.get(id(current))
        for scope, position in local_bindings.get(name, ()):
            if id(scope) in ancestors and (
                position is None or child.position is None or position <= child.position
            ):
                return True
        return False

    for _, child in nodes:
        if isinstance(child, jt.VariableDeclaration) and not isinstance(child, jt.FieldDeclaration):
            for item in child.declarators:
                bind(item.name, child, position=child.position)
        elif isinstance(child, (jt.FormalParameter, jt.InferredFormalParameter, jt.CatchClauseParameter, jt.TryResource)):
            bind(child.name, child)
        elif isinstance(child, jt.LambdaExpression):
            for parameter in child.parameters:
                bind(getattr(parameter, "name", None) or getattr(parameter, "member", ""), child, scope=child)
        elif isinstance(child, jt.TypeParameter):
            type_parameters.add(child.name)

    def add(value, role):
        if not isinstance(value, str) or value in type_parameters:
            return
        value = re.sub(r"(?:\[\])+$", "", value)
        if is_entity_candidate(value, role=role):
            entities.add(value)
            # Keep a useful short alias, never each package segment.
            terminal = value.rsplit(".", 1)[-1]
            if is_entity_candidate(terminal, role=role):
                entities.add(terminal)

    for parent, child in nodes:
        if isinstance(child, jt.ReferenceType):
            if not isinstance(parent, jt.ReferenceType) or parent.sub_type is not child:
                add(type_name(child), "type")
        elif isinstance(child, jt.Annotation):
            add(child.name, "type")
        elif isinstance(child, (jt.MethodDeclaration, jt.ConstructorDeclaration)):
            add(child.name, "method")
            for exception_type in child.throws or []:
                add(exception_type, "type")
        elif isinstance(child, jt.CatchClauseParameter):
            for exception_type in child.types:
                add(exception_type, "type")
        elif isinstance(child, jt.FieldDeclaration):
            for declarator in child.declarators:
                add(declarator.name, "field")
        elif isinstance(child, (jt.MethodInvocation, jt.SuperMethodInvocation)):
            add(child.member, "method")
            qualifier = getattr(child, "qualifier", "") or ""
            if qualifier in field_names and not locally_bound(qualifier, child):
                add(qualifier, "field")
            elif qualifier and any(part[:1].isupper() for part in qualifier.split(".")):
                entities.update(code_identifiers(qualifier))
        elif isinstance(child, (jt.MemberReference, jt.SuperMemberReference)):
            # Local/parameter reads do not become cross-document graph anchors.
            qualifier = getattr(child, "qualifier", "") or ""
            explicit_field = bool(qualifier) or isinstance(parent, jt.This) or isinstance(child, jt.SuperMemberReference)
            if explicit_field or child.member not in local_names or (
                child.member in field_names and not locally_bound(child.member, child)
            ):
                add(child.member, "field")
            if qualifier and any(part[:1].isupper() for part in qualifier.split(".")):
                entities.update(code_identifiers(qualifier))
        elif isinstance(child, jt.Literal) and child.value.startswith('"'):
            for value in code_identifiers(child.value[1:-1]):
                if value.isupper() or re.search(r"\d", value) or value.endswith(("Exception", "Error")):
                    entities.add(value)
    return entities


def method_scope_nodes(node, parents=()):
    """Walk one method, retaining creator arguments but excluding nested type bodies."""
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from method_scope_nodes(child, parents)
    elif isinstance(node, javalang.ast.Node) and not isinstance(node, jt.TypeDeclaration):
        yield parents, node
        for attribute in node.attrs:
            if isinstance(node, jt.Creator) and attribute == "body":
                continue
            yield from method_scope_nodes(getattr(node, attribute), (*parents, node))


@dataclass
class JavaFile:
    passages: list[Passage] = field(default_factory=list)
    classes: dict = field(default_factory=dict)
    methods: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_java(project: str, path: str, source: str) -> JavaFile:
    result = JavaFile()
    lines = source.splitlines()

    def chunk(start, end, kind, owner="", symbol="", entities=None, structured=True):
        text = "\n".join(lines[start - 1 : end])
        result.passages.append(
            Passage(
                id=digest(f"{path}:{start}:{end}:{kind}:{symbol}"),
                project=project,
                file_path=path,
                class_name=owner,
                symbol=symbol,
                kind=kind,
                start_line=start,
                end_line=end,
                content=text,
                content_hash=digest(text),
                entities=sorted(set(code_identifiers(text) if entities is None else entities)),
                structured=structured,
            )
        )

    try:
        tokens = list(javalang.tokenizer.tokenize(source))
        unit = javalang.parser.Parser(tokens).parse()
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, TypeError, IndexError) as exc:
        result.warnings.append(f"{path}: Java 8 解析失败，使用文本分块 ({type(exc).__name__})")
        for start in range(1, len(lines) + 1, 100):
            chunk(start, min(start + 99, len(lines)), "text", structured=False)
        return result

    package = unit.package.name if unit.package else ""
    imports = [i.path + (".*" if i.wildcard else "") for i in unit.imports if not i.static]
    static_imports = [i.path + (".*" if i.wildcard else "") for i in unit.imports if i.static]

    token_indices = {token.position: index for index, token in enumerate(tokens)}
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    def offset(token):
        return line_offsets[token.position.line - 1] + token.position.column - 1

    def declaration_start(node):
        positions = [node.position] + [a.position for a in getattr(node, "annotations", []) if a.position]
        index = token_indices[min(positions)]
        while index and isinstance(tokens[index - 1], javalang.tokenizer.Modifier):
            index -= 1
        start = tokens[index].position.line
        documentation = getattr(node, "documentation", None)
        if documentation:
            previous_end = offset(tokens[index - 1]) + len(tokens[index - 1].value) if index else 0
            # Match only the gap before this declaration, never an earlier member's text.
            doc_offset = source.rfind(documentation, previous_end, offset(tokens[index]))
            if doc_offset >= 0:
                start = source.count("\n", 0, doc_offset) + 1
        return start

    def bounds(node):
        start = declaration_start(node)
        index = token_indices[node.position]
        # Annotation arguments and method parameters may themselves contain braces.
        parens = brackets = depth = 0
        for token in tokens[index:]:
            value = token.value if isinstance(token, javalang.tokenizer.Separator) else ""
            if value == "(":
                parens += 1
            elif value == ")":
                parens -= 1
            elif value == "[":
                brackets += 1
            elif value == "]":
                brackets -= 1
            elif value == "{" and (depth or (parens == 0 and brackets == 0)):
                depth += 1
            elif value == "}" and depth:
                depth -= 1
                if depth == 0 and not isinstance(node, jt.FieldDeclaration):
                    return start, token.position.line
            elif value == ";" and depth == 0 and parens == 0 and brackets == 0:
                return start, token.position.line
        return start, len(lines)

    types = (jt.ClassDeclaration, jt.InterfaceDeclaration, jt.EnumDeclaration, jt.AnnotationDeclaration)
    for ancestry, decl in unit:
        if not isinstance(decl, types) or not decl.position:
            continue
        # Local/anonymous classes need additional scope resolution; retain them as text inside their method.
        if any(
            isinstance(
                a, (jt.MethodDeclaration, jt.ConstructorDeclaration, jt.Creator, jt.EnumConstantDeclaration)
            )
            for a in ancestry
        ):
            continue
        outer = [a.name for a in ancestry if isinstance(a, types)]
        owner = ".".join(filter(None, [package, *outer, decl.name]))
        members = decl.body.declarations if isinstance(decl.body, jt.EnumBody) else decl.body
        members = members or []
        methods = [n for n in members if isinstance(n, (jt.MethodDeclaration, jt.ConstructorDeclaration))]
        fields = {}
        for f in members:
            if isinstance(f, jt.FieldDeclaration):
                fields.update(
                    {d.name: type_name(f.type) + "[]" * len(d.dimensions or []) for d in f.declarators}
                )
        result.classes[owner] = {
            "name": decl.name,
            "package": package,
            "imports": imports,
            "static_imports": static_imports,
            "fields": fields,
            "interface": isinstance(decl, jt.InterfaceDeclaration),
            "final": "final" in decl.modifiers,
            "extends": bool(getattr(decl, "extends", None)),
        }
        start, end = bounds(decl)
        positioned = [declaration_start(n) for n in members if getattr(n, "position", None)]
        header_end = min(positioned) - 1 if positioned else end
        owner_entities = {owner, decl.name}
        header_entities = ast_entities([
            getattr(decl, "extends", None), getattr(decl, "implements", None),
            getattr(decl, "annotations", None), getattr(decl, "type_parameters", None),
        ])
        chunk(start, max(start, header_end), "class", owner, entities=owner_entities | header_entities)
        for f in members:
            if isinstance(f, jt.FieldDeclaration) and f.position:
                a, b = bounds(f)
                chunk(a, b, "field", owner, entities=ast_entities(f, fields) | owner_entities)
            elif isinstance(f, list) and f:
                # javalang represents static/instance initializer blocks as statement lists.
                positions = [n.position for statement in f for _, n in statement if n.position]
                if positions:
                    first = min(positions)
                    before = [
                        i
                        for i, t in enumerate(tokens)
                        if t.position < first
                        and isinstance(t, javalang.tokenizer.Separator)
                        and t.value == "{"
                    ]
                    if before:
                        opening = before[-1]
                        depth = 0
                        for token in tokens[opening:]:
                            if isinstance(token, javalang.tokenizer.Separator):
                                depth += (token.value == "{") - (token.value == "}")
                                if depth == 0:
                                    chunk(
                                        tokens[opening].position.line,
                                        token.position.line,
                                        "initializer",
                                        owner,
                                        entities=ast_entities(f, fields) | owner_entities,
                                    )
                                    break
        for method in methods:
            if not method.position:
                continue
            a, b = bounds(method)
            params = [type_name(p.type) + ("[]" if p.varargs else "") for p in method.parameters]
            symbol = f"{owner}#{method.name}({','.join(params)})"
            entities = ast_entities(method, fields) | owner_entities | {symbol}
            chunk(
                a,
                b,
                "constructor" if isinstance(method, jt.ConstructorDeclaration) else "method",
                owner,
                symbol,
                entities,
            )
            result.methods.append(
                {
                    "symbol": symbol,
                    "owner": owner,
                    "name": method.name,
                    "arity": len(params),
                    "params": params,
                    "file_path": path,
                    "start_line": a,
                    "end_line": b,
                    "modifiers": sorted(method.modifiers),
                    "varargs": any(p.varargs for p in method.parameters),
                }
            )
            variables = dict(fields)
            variables.update({p.name: type_name(p.type) for p in method.parameters})
            initial_variables = dict(variables)
            binding_positions = {}
            ambiguous_variables = set()
            scoped_nodes = list(method_scope_nodes(method))
            # Without a full symbol table, nested lexical bindings remain conservative.
            lexical_scopes = (
                jt.BlockStatement,
                jt.ForStatement,
                jt.CatchClause,
                jt.LambdaExpression,
                jt.TryStatement,
                jt.SwitchStatement,
                jt.SynchronizedStatement,
            )
            for local_path, local in scoped_nodes:
                bindings = []
                if isinstance(local, jt.VariableDeclaration):
                    bindings = [
                        (v.name, type_name(local.type) + "[]" * len(v.dimensions or []))
                        for v in local.declarators
                    ]
                elif isinstance(local, (jt.CatchClauseParameter, jt.TryResource, jt.InferredFormalParameter)):
                    bindings = [(local.name, "")]
                elif isinstance(local, jt.LambdaExpression):
                    bindings = [
                        (getattr(p, "name", None) or getattr(p, "member", ""), "") for p in local.parameters
                    ]
                nested_scope = any(isinstance(n, lexical_scopes) for n in local_path)
                for name, local_type in bindings:
                    if not name:
                        continue
                    if nested_scope or not local_type or name in variables and variables[name] != local_type:
                        ambiguous_variables.add(name)
                    variables[name] = local_type
                    if local.position:
                        binding_positions[name] = local.position
            for call_path, call in scoped_nodes:
                if not isinstance(call, jt.MethodInvocation) or not call.position:
                    continue
                qualifier = call.qualifier or ""
                selector_owner = next(
                    (
                        n
                        for n in reversed(call_path)
                        if any(s is call for s in (getattr(n, "selectors", None) or []))
                    ),
                    None,
                )
                explicit_this = (
                    isinstance(selector_owner, jt.This)
                    and not selector_owner.qualifier
                    and selector_owner.selectors[0] is call
                )
                chained = selector_owner is not None and not explicit_this
                receiver_type = variables.get(qualifier, "")
                if qualifier in binding_positions and call.position < binding_positions[qualifier]:
                    receiver_type = initial_variables.get(qualifier, "")
                result.calls.append(
                    {
                        "caller": symbol,
                        "owner": owner,
                        "name": call.member,
                        "receiver": "this" if explicit_this else qualifier,
                        "receiver_type": receiver_type,
                        "arity": len(call.arguments),
                        "file_path": path,
                        "line": call.position.line,
                        "chained": chained,
                        "ambiguous_receiver": qualifier in ambiguous_variables,
                    }
                )
    return result


def resolve_calls(classes: dict, methods: list[dict], calls: list[dict]) -> list[dict]:
    """Conservative: never equate entity co-occurrence with a resolved call."""
    output = []
    for call in calls:
        info = classes[call["owner"]]
        receiver = call["receiver"]
        owner_candidates = []
        reason = ""
        if call.get("ambiguous_receiver"):
            reason = "接收者存在局部变量遮蔽，作用域类型无法唯一确定"
        elif call["chained"]:
            reason = "链式调用的接收者类型未知"
        elif receiver in ("", "this"):
            owner_candidates = [call["owner"]]
            if not receiver:
                imported_owners = {
                    entry.rsplit(".", 1)[0]
                    for entry in info.get("static_imports", [])
                    if entry.endswith(".*") or entry.endswith("." + call["name"])
                }
                if imported_owners:
                    owner_candidates += sorted(imported_owners.intersection(classes))
                    owner_candidates = sorted(set(owner_candidates))
                    reason = "静态导入目标需结合成员遮蔽和实参类型确认"
        else:
            type_hint = call["receiver_type"] or receiver
            candidates = [type_hint, info["package"] + "." + type_hint]
            candidates += [i for i in info["imports"] if i.endswith("." + type_hint)]
            candidates += [i[:-1] + type_hint for i in info["imports"] if i.endswith(".*")]
            owner_candidates = sorted({c for c in candidates if c in classes})
        candidates = [
            m
            for m in methods
            if m["name"] == call["name"]
            and (m["arity"] == call["arity"] or m["varargs"])
            and (not owner_candidates or m["owner"] in owner_candidates)
        ]
        if len(owner_candidates) != 1:
            reason = reason or "无法唯一解析接收者（外部依赖、导入、反射或静态导入）"
        elif len(candidates) != 1:
            reason = reason or "重载或继承目标不唯一，需要实参类型/运行时信息"
        else:
            target = candidates[0]
            cls = classes[target["owner"]]
            if cls["interface"] or (
                not cls["final"] and not {"static", "private", "final"}.intersection(target["modifiers"])
            ):
                reason = reason or "接口/可覆盖方法可能存在动态分派或依赖注入"
            elif cls["extends"] and receiver in ("", "this"):
                reason = reason or "继承关系未完全解析"
        if info.get("ambiguous_definition") or any(
            classes[m["owner"]].get("ambiguous_definition") for m in candidates
        ):
            reason = "存在跨文件同名类，无法确定所属模块和实际调用目标"
        output.append(
            {
                **call,
                "target": candidates[0]["symbol"] if not reason else None,
                "candidates": [m["symbol"] for m in candidates],
                "unknown_reason": reason,
            }
        )
    return output
