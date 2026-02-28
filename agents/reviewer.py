"""
reviewer.py
─────────────────────────────────────────────────────────────────────────────
Agente Code Reviewer do IT Department Multi-Agent System.

Responsabilidades:
  • Analisar diffs e pull requests
  • Detectar code smells e anti-patterns
  • Scan de vulnerabilidades de segurança
  • Medir complexidade ciclomática
  • Verificar boas práticas (SOLID, DRY, KISS)
  • Emitir veredicto: APROVADO / PRECISA DE AJUSTES / REPROVADO

Ferramentas:
  ast_tools  → analyze_complexity, find_code_smells, check_security
  git_tools  → git_diff, git_log, git_blame, git_show_commit
  fs_tools   → read_file, search_in_files, get_file_info, get_repo_tree
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

from llm_factory import make_llm
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from state import AgentState
from supervisor import record_agent_output
from tools.filesystem import REVIEWER_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

REVIEWER_MODEL = os.environ.get("ITDEPT_REVIEWER_MODEL", "claude-sonnet-4-5")

try:
    from tools.filesystem import ALLOWED_BASE_PATH
except ImportError:
    ALLOWED_BASE_PATH = Path(os.environ.get("ITDEPT_BASE_PATH", str(Path.cwd()))).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

REVIEWER_SYSTEM_PROMPT = """\
Você é um Code Reviewer sênior com 10+ anos de experiência em Python.
Seu papel é garantir que o código que entra no repositório seja seguro,
manutenível e siga boas práticas de engenharia de software.

## SUAS FERRAMENTAS

- analyze_complexity  → mede complexidade ciclomática de funções
- find_code_smells    → detecta anti-patterns comuns
- check_security      → identifica vulnerabilidades de segurança
- read_file           → leia o código para análise detalhada
- search_in_files     → busque padrões específicos no repositório
- get_repo_tree       → entenda a estrutura do projeto
- get_file_info       → metadata de arquivos
- git_diff            → veja as mudanças recentes
- git_log             → entenda o histórico
- git_blame           → veja quem escreveu cada linha
- git_show_commit     → detalhes de um commit específico

## PROCESSO DE REVISÃO

1. **Entenda o contexto**
   - Use git_diff para ver o que mudou
   - Leia os arquivos modificados com read_file
   - Verifique o histórico com git_log se necessário

2. **Análise estática**
   - analyze_complexity nos arquivos modificados
   - find_code_smells para anti-patterns
   - check_security para vulnerabilidades

3. **Revisão manual**
   - Lógica correta e edge cases tratados?
   - Nomes claros e autodocumentados?
   - Princípios SOLID respeitados?
   - DRY: código duplicado?
   - Error handling adequado?
   - Logs e observabilidade presentes?

4. **Emita veredicto**

## CRITÉRIOS DE VEREDICTO

✅ APROVADO — pode avançar para docs/deploy
   - Sem vulnerabilidades críticas
   - Complexidade ciclomática ≤ 10 por função
   - Sem code smells críticos
   - Lógica correta e tratamento de erros adequado

⚠️ PRECISA DE AJUSTES — volta para o Developer
   - Problemas menores que devem ser corrigidos
   - Sugestões de melhoria importantes
   - Complexidade alta mas não bloqueante

❌ REPROVADO — volta imediatamente para o Developer
   - Vulnerabilidade de segurança crítica
   - Bug óbvio que quebraria produção
   - Complexidade ciclomática > 20
   - Código ilegível ou sem estrutura

## FORMATO DO REPORT

```
## Veredicto: [✅ APROVADO | ⚠️ PRECISA DE AJUSTES | ❌ REPROVADO]

## Resumo
<2-3 linhas descrevendo o que foi revisado>

## Problemas Críticos (bloqueantes)
- [arquivo:linha] descrição do problema

## Sugestões de Melhoria
- [arquivo:linha] sugestão

## Pontos Positivos
- o que foi bem feito

## Métricas
- Complexidade máxima: X (função Y)
- Vulnerabilidades: X críticas, X avisos
- Code smells: X
```

Seja direto, construtivo e específico. Aponte linha e arquivo sempre que possível.
"""

# ─────────────────────────────────────────────────────────────────────────────
# AST Analysis Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def analyze_complexity(
    path: str,
    max_complexity: int = 10,
    repo_path: Optional[str] = None,
) -> str:
    """
    Analisa a complexidade ciclomática de todas as funções/métodos de um arquivo Python.
    Complexidade > 10 é difícil de testar. > 20 é crítico.

    Args:
        path:           Caminho do arquivo Python relativo ao workspace.
        max_complexity: Threshold para marcar como complexo (padrão: 10).
        repo_path:      Diretório raiz do repositório.

    Returns:
        Relatório de complexidade por função, ordenado do mais ao menos complexo.
    """
    cwd      = Path(repo_path or ALLOWED_BASE_PATH)
    filepath = (cwd / path).resolve()

    if not filepath.exists():
        return f"[ERRO] Arquivo não encontrado: '{path}'"
    if filepath.suffix != ".py":
        return f"[AVISO] analyze_complexity funciona apenas com arquivos .py"

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree   = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return f"[ERRO] Sintaxe inválida em '{path}': {e}"

    results: list[tuple[int, str, int]] = []  # (complexity, name, lineno)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = _calc_complexity(node)
            results.append((complexity, node.name, node.lineno))

    if not results:
        return f"[INFO] Nenhuma função encontrada em '{path}'"

    results.sort(reverse=True)

    lines = [f"📊 Complexidade ciclomática — {path}"]
    lines.append("─" * 50)

    critical = [r for r in results if r[0] > 20]
    high     = [r for r in results if max_complexity < r[0] <= 20]
    ok       = [r for r in results if r[0] <= max_complexity]

    if critical:
        lines.append("❌ CRÍTICO (> 20):")
        for c, name, line in critical:
            lines.append(f"   {c:3d}  {name}()  [linha {line}]")

    if high:
        lines.append(f"⚠️  ALTO ({max_complexity+1}–20):")
        for c, name, line in high:
            lines.append(f"   {c:3d}  {name}()  [linha {line}]")

    if ok:
        lines.append(f"✅ Aceitável (≤ {max_complexity}):")
        for c, name, line in ok[:10]:  # Mostra até 10
            lines.append(f"   {c:3d}  {name}()  [linha {line}]")
        if len(ok) > 10:
            lines.append(f"   ... e mais {len(ok)-10} funções dentro do limite")

    max_c = results[0][0] if results else 0
    lines.append(f"\nResumo: {len(results)} funções | máx: {max_c} | "
                 f"críticas: {len(critical)} | altas: {len(high)}")

    return "\n".join(lines)


def _calc_complexity(node: ast.AST) -> int:
    """Calcula complexidade ciclomática de um nó AST (McCabe simplificado)."""
    complexity = 1
    branch_nodes = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
        ast.BoolOp,  # and/or também ramificam
    )
    for child in ast.walk(node):
        if isinstance(child, branch_nodes):
            complexity += 1
        # Ternários também aumentam complexidade
        elif isinstance(child, ast.IfExp):
            complexity += 1
    return complexity


@tool
def find_code_smells(
    path: str,
    repo_path: Optional[str] = None,
) -> str:
    """
    Detecta anti-patterns e code smells comuns em código Python.

    Detecta: funções longas, muitos parâmetros, god classes, magic numbers,
    bare except, print statements, TODO/FIXME, imports wildcard, variáveis
    de uma letra, e outros.

    Args:
        path:      Arquivo Python para analisar.
        repo_path: Diretório raiz do repositório.

    Returns:
        Lista de smells encontrados com localização e severidade.
    """
    cwd      = Path(repo_path or ALLOWED_BASE_PATH)
    filepath = (cwd / path).resolve()

    if not filepath.exists():
        return f"[ERRO] Arquivo não encontrado: '{path}'"

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree   = ast.parse(source, filename=str(filepath))
        lines  = source.splitlines()
    except SyntaxError as e:
        return f"[ERRO] Sintaxe inválida: {e}"

    smells: list[tuple[str, int, str, str]] = []  # (severity, lineno, smell, detail)

    for node in ast.walk(tree):

        # Funções muito longas (> 50 linhas)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", node.lineno)
            length   = end_line - node.lineno
            if length > 50:
                smells.append(("⚠️", node.lineno, "Função longa",
                               f"'{node.name}' tem {length} linhas (máx recomendado: 50)"))

            # Muitos parâmetros (> 5)
            n_args = len(node.args.args)
            if n_args > 5:
                smells.append(("⚠️", node.lineno, "Muitos parâmetros",
                               f"'{node.name}' tem {n_args} parâmetros (máx recomendado: 5)"))

        # Classes muito grandes (> 300 linhas ou > 20 métodos)
        if isinstance(node, ast.ClassDef):
            end_line = getattr(node, "end_lineno", node.lineno)
            methods  = [n for n in ast.walk(node)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if end_line - node.lineno > 300:
                smells.append(("⚠️", node.lineno, "God Class",
                               f"'{node.name}' tem {end_line - node.lineno} linhas"))
            if len(methods) > 20:
                smells.append(("⚠️", node.lineno, "God Class",
                               f"'{node.name}' tem {len(methods)} métodos"))

        # Bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            smells.append(("❌", node.lineno, "Bare except",
                           "Captura todas as exceções sem especificar o tipo"))

        # Import wildcard
        if isinstance(node, ast.ImportFrom) and any(
            isinstance(a, ast.alias) and a.name == "*" for a in node.names
        ):
            smells.append(("⚠️", node.lineno, "Wildcard import",
                           f"'from {node.module} import *' polui o namespace"))

    # Análise de texto linha por linha
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # print() em código (exceto em __main__ e tests)
        if re.match(r'\bprint\s*\(', stripped) and "test" not in path.lower():
            smells.append(("ℹ️", i, "print() statement",
                           "Use logging ao invés de print() em produção"))

        # Magic numbers
        magic = re.findall(r'(?<!["\'\w])\b(?!0\b|1\b)(\d{2,})\b(?!["\'\w])', stripped)
        if magic and not stripped.startswith("#"):
            smells.append(("ℹ️", i, "Magic number",
                           f"Considere extrair {magic[0]} para uma constante nomeada"))

        # TODO / FIXME / HACK
        if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', stripped, re.IGNORECASE):
            tag = re.search(r'\b(TODO|FIXME|HACK|XXX)\b', stripped, re.IGNORECASE).group()
            smells.append(("ℹ️", i, f"{tag} pendente",
                           stripped[:80]))

        # Variáveis de uma letra fora de loops
        if re.match(r'^\s*([a-zA-Z])\s*=\s*(?!range)', line) and \
           not re.search(r'\bfor\b', line):
            var = re.match(r'^\s*([a-zA-Z])\s*=', line).group(1)
            if var not in ("i", "j", "k", "x", "y", "z", "n", "_"):
                smells.append(("ℹ️", i, "Nome pouco descritivo",
                               f"Variável '{var}' — use nomes mais descritivos"))

    if not smells:
        return f"✅ Nenhum code smell detectado em '{path}'"

    smells.sort(key=lambda s: ({"❌": 0, "⚠️": 1, "ℹ️": 2}.get(s[0], 3), s[1]))

    critical = [s for s in smells if s[0] == "❌"]
    warnings = [s for s in smells if s[0] == "⚠️"]
    infos    = [s for s in smells if s[0] == "ℹ️"]

    output_lines = [
        f"🔍 Code smells em '{path}' — "
        f"{len(critical)} críticos | {len(warnings)} avisos | {len(infos)} infos",
        "─" * 55,
    ]
    for severity, lineno, smell, detail in smells[:30]:
        output_lines.append(f"  {severity}  linha {lineno:4d}  [{smell}]  {detail}")

    if len(smells) > 30:
        output_lines.append(f"\n  ... e mais {len(smells)-30} ocorrências")

    return "\n".join(output_lines)


@tool
def check_security(
    path: str,
    repo_path: Optional[str] = None,
) -> str:
    """
    Verifica vulnerabilidades de segurança comuns em código Python.

    Detecta: SQL injection, command injection, hardcoded secrets,
    uso de eval/exec, deserialização insegura, weak crypto, etc.

    Args:
        path:      Arquivo Python para analisar.
        repo_path: Diretório raiz do repositório.

    Returns:
        Lista de vulnerabilidades encontradas com severidade e localização.
    """
    cwd      = Path(repo_path or ALLOWED_BASE_PATH)
    filepath = (cwd / path).resolve()

    if not filepath.exists():
        return f"[ERRO] Arquivo não encontrado: '{path}'"

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        lines  = source.splitlines()
    except Exception as e:
        return f"[ERRO] Não foi possível ler '{path}': {e}"

    vulns: list[tuple[str, int, str, str]] = []  # (severity, lineno, vuln, detail)

    # Padrões de vulnerabilidades
    patterns = [
        # (severity, regex, nome, descrição)
        ("🔴 CRÍTICO", r'\beval\s*\(',       "eval()",         "Execução de código arbitrário"),
        ("🔴 CRÍTICO", r'\bexec\s*\(',       "exec()",         "Execução de código arbitrário"),
        ("🔴 CRÍTICO", r'pickle\.loads?\(',   "pickle.load",    "Deserialização insegura — RCE"),
        ("🔴 CRÍTICO", r'subprocess.*shell\s*=\s*True', "shell=True", "Command injection via shell=True"),
        ("🔴 CRÍTICO", r'os\.system\s*\(',   "os.system()",    "Command injection"),
        ("🟠 ALTO",    r'yaml\.load\s*\(',   "yaml.load()",    "Use yaml.safe_load() em vez disso"),
        ("🟠 ALTO",    r'hashlib\.md5\(',    "MD5",            "Hash fraco — use SHA-256+"),
        ("🟠 ALTO",    r'hashlib\.sha1\(',   "SHA-1",          "Hash fraco — use SHA-256+"),
        ("🟠 ALTO",    r'random\.',          "random module",  "Use secrets para dados criptográficos"),
        ("🟡 MÉDIO",   r'assert\s+',         "assert",         "Assertions removidas com -O — não use para validação"),
        ("🟡 MÉDIO",   r'DEBUG\s*=\s*True',  "DEBUG=True",     "Debug ativado — não vá para produção assim"),
        ("🟡 MÉDIO",   r'ALLOWED_HOSTS\s*=\s*\[.*\*', "ALLOWED_HOSTS=*", "Host wildcard em produção"),
        ("🟡 MÉDIO",   r'verify\s*=\s*False', "SSL verify=False", "Verificação SSL desabilitada"),
    ]

    # Padrões de segredos hardcoded
    secret_patterns = [
        (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']',  "Senha hardcoded"),
        (r'(?i)(api_key|apikey|api_secret)\s*=\s*["\'][^"\']{8,}["\']', "API key hardcoded"),
        (r'(?i)(secret_key|secret)\s*=\s*["\'][^"\']{8,}["\']',    "Secret key hardcoded"),
        (r'(?i)(token)\s*=\s*["\'][a-zA-Z0-9._-]{20,}["\']',       "Token hardcoded"),
        (r'(?i)(aws_access_key|aws_secret)\s*=\s*["\'][^"\']+["\']', "AWS credential hardcoded"),
    ]

    for i, line in enumerate(lines, 1):
        # Ignora comentários e docstrings simples
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for severity, pattern, name, desc in patterns:
            if re.search(pattern, line):
                vulns.append((severity, i, name, desc))

        for pattern, name in secret_patterns:
            if re.search(pattern, line):
                vulns.append(("🔴 CRÍTICO", i, name, line.strip()[:60]))

    # Tenta usar bandit se disponível (mais completo)
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "bandit", "-r", str(filepath), "-f", "text", "-ll"],
            capture_output=True, text=True, timeout=30,
            cwd=str(ALLOWED_BASE_PATH),
        )
        if result.returncode == 0 and result.stdout:
            bandit_section = f"\n\n📦 Análise bandit:\n{result.stdout[:800]}"
        else:
            bandit_section = ""
    except Exception:
        bandit_section = ""

    if not vulns:
        base = f"✅ Nenhuma vulnerabilidade detectada em '{path}'"
        return base + bandit_section

    vulns.sort(key=lambda v: ({"🔴 CRÍTICO": 0, "🟠 ALTO": 1, "🟡 MÉDIO": 2}.get(v[0], 3), v[1]))

    output_lines = [
        f"🛡️  Segurança — '{path}'",
        f"   {sum(1 for v in vulns if '🔴' in v[0])} críticos | "
        f"{sum(1 for v in vulns if '🟠' in v[0])} altos | "
        f"{sum(1 for v in vulns if '🟡' in v[0])} médios",
        "─" * 55,
    ]
    for severity, lineno, name, detail in vulns:
        output_lines.append(f"  {severity}  linha {lineno:4d}  [{name}]")
        output_lines.append(f"             {detail}")

    return "\n".join(output_lines) + bandit_section


# ─────────────────────────────────────────────────────────────────────────────
# Tools do Reviewer
# ─────────────────────────────────────────────────────────────────────────────

REVIEWER_AST_TOOLS = [analyze_complexity, find_code_smells, check_security]

try:
    from tools.git_tools import REVIEWER_GIT_TOOLS
except ImportError:
    REVIEWER_GIT_TOOLS = []

ALL_REVIEWER_TOOLS = REVIEWER_AST_TOOLS + REVIEWER_GIT_TOOLS + REVIEWER_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Construção do agente
# ─────────────────────────────────────────────────────────────────────────────

_reviewer_agent_instance = None

def _get_reviewer_agent():
    global _reviewer_agent_instance
    if _reviewer_agent_instance is None:
        llm = make_llm("reviewer", temperature=0, max_tokens=4096)
        _reviewer_agent_instance = create_react_agent(
            model=llm,
            tools=ALL_REVIEWER_TOOLS,
            state_modifier=SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        )
    return _reviewer_agent_instance


# ─────────────────────────────────────────────────────────────────────────────
# Nó do grafo
# ─────────────────────────────────────────────────────────────────────────────

def reviewer_node(state: AgentState) -> AgentState:
    """
    Nó do Code Reviewer Agent no grafo LangGraph.

    Analisa o código modificado e emite um veredicto estruturado
    que o supervisor usa para decidir: aprovar, ajustar ou reprovar.
    """
    instruction   = state.get("current_instruction", "")
    repo_path     = state.get("repo_path", ".")
    task          = state.get("task", "")
    changed_files = state.get("artifacts", {}).get("files_changed", [])

    changed_ctx = ""
    if changed_files:
        changed_ctx = (
            "\n\n## Arquivos que foram modificados:\n"
            + "\n".join(f"  - {f}" for f in changed_files)
        )

    user_prompt = f"""\
## TASK ORIGINAL
{task}

## SUA INSTRUÇÃO (do IT Manager)
{instruction}
{changed_ctx}

## REPOSITÓRIO
{repo_path}

Execute a revisão completa:
1. Use git_diff para ver todas as mudanças
2. Rode analyze_complexity nos arquivos modificados
3. Rode find_code_smells em cada arquivo alterado
4. Rode check_security nos arquivos críticos
5. Leia os arquivos com read_file para revisão manual

Emita o veredicto final usando o formato do report.
"""

    try:
        agent  = _get_reviewer_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        final_message = result["messages"][-1]
        output = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )

        status, verdict = _infer_review_verdict(output)
        artifacts = {"review_verdict": verdict, "review_done": True}

    except Exception as e:
        output    = f"❌ Erro no Reviewer Agent: {type(e).__name__}: {e}"
        status    = "error"
        artifacts = {"review_verdict": "ERROR", "review_done": False}

    updates = record_agent_output(
        state=state,
        agent_name="reviewer",
        output=output,
        status=status,
        artifacts=artifacts,
    )
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_review_verdict(output: str) -> tuple[str, str]:
    """
    Extrai o veredicto do report do reviewer.
    Retorna (status_para_record_output, verdict_string).
    """
    lower = output.lower()

    if "reprovado" in lower or "❌" in output and "reprovado" in lower:
        return "error", "REPROVADO"

    if "precisa de ajustes" in lower or "⚠️" in output and "ajustes" in lower:
        return "warning", "PRECISA_AJUSTES"

    if "aprovado" in lower or "✅" in output and "aprovado" in lower:
        return "success", "APROVADO"

    # Sem veredicto explícito — assume warning para ser conservador
    return "warning", "INCONCLUSIVO"


# ─────────────────────────────────────────────────────────────────────────────
# Exportações
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["reviewer_node", "REVIEWER_AST_TOOLS", "ALL_REVIEWER_TOOLS"]