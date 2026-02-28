"""
qa.py
─────────────────────────────────────────────────────────────────────────────
Agente QA Engineer do IT Department Multi-Agent System.

Responsabilidades:
  • Executar testes automatizados (pytest)
  • Medir cobertura de código (coverage.py)
  • Rodar linters (ruff, pylint)
  • Type checking (mypy)
  • Gerar relatório de qualidade estruturado
  • Criar testes unitários para código sem cobertura

Ferramentas:
  shell  → run_pytest, run_coverage, run_linter, run_type_check, run_command
  fs     → read_file, write_file, search_in_files, list_directory
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from llm_factory import make_llm
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from state import AgentState
from supervisor import record_agent_output
from tools.filesystem import QA_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

QA_MODEL   = os.environ.get("ITDEPT_QA_MODEL", "claude-sonnet-4-5")
QA_TIMEOUT = int(os.environ.get("ITDEPT_QA_TIMEOUT", "120"))

try:
    from tools.filesystem import ALLOWED_BASE_PATH
except ImportError:
    ALLOWED_BASE_PATH = Path(os.environ.get("ITDEPT_BASE_PATH", str(Path.cwd()))).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """\
Você é um QA Engineer sênior especializado em qualidade de software Python.
Sua missão é garantir que o código é correto, limpo e bem coberto por testes.

## SUAS FERRAMENTAS

- run_pytest       → executa a suíte de testes
- run_coverage     → mede cobertura de código
- run_linter       → ruff ou pylint para style/erros
- run_type_check   → mypy para verificação de tipos
- run_command      → comandos shell genéricos (use com cuidado)
- read_file        → leia arquivos para entender o contexto
- write_file       → crie novos arquivos de teste
- search_in_files  → busque padrões no código
- list_directory   → explore a estrutura do projeto

## PROCESSO DE TRABALHO

1. **Descubra a estrutura de testes**
   - Use list_directory para localizar pasta de testes (tests/, test/, src/tests/)
   - Use search_in_files para encontrar arquivos de teste existentes

2. **Execute a suíte completa**
   - run_pytest no diretório de testes
   - Se falhar, analise os erros e identifique a causa raiz

3. **Meça cobertura**
   - run_coverage para ver quais linhas não são testadas
   - Foco em módulos que sofreram mudanças recentes

4. **Rode linter e type check**
   - run_linter nos arquivos modificados
   - run_type_check para detectar erros de tipo

5. **Crie testes faltantes (se instruído)**
   - Use read_file para entender a função/classe a testar
   - Escreva testes com pytest: casos normais, edge cases, erros esperados
   - Salve em tests/test_<modulo>.py

## BOAS PRÁTICAS DE TESTES

- Um teste deve testar UMA coisa só
- Use fixtures do pytest para setup/teardown
- Nomeie: test_<função>_<cenário>_<resultado_esperado>
- Mock dependências externas (IO, HTTP, banco)
- Cubra: happy path, edge cases, exceções esperadas

## REPORT FINAL

Estruture sempre assim:
```
## Resultado dos Testes
- Total: X | Passou: X | Falhou: X | Erros: X

## Cobertura
- Global: X%
- Arquivos críticos: <lista>

## Linting
- Erros: X | Avisos: X
- Problemas críticos: <lista>

## Type Check
- Erros: X
- Problemas: <lista>

## Ações Tomadas
- <o que foi feito>

## Recomendações
- <o que o Developer deve corrigir>
```

O supervisor usa este report para decidir se manda de volta ao Developer
ou avança para o Code Reviewer. Seja preciso e objetivo.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Shell Tools (específicas do QA)
# ─────────────────────────────────────────────────────────────────────────────

def _run_shell(cmd: list[str], cwd: Optional[str] = None, timeout: int = QA_TIMEOUT) -> tuple[str, str, int]:
    """Executa um comando shell e retorna (stdout, stderr, returncode)."""
    workdir = cwd or str(ALLOWED_BASE_PATH)
    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError as e:
        return "", f"Comando não encontrado: {e}", 127
    except subprocess.TimeoutExpired:
        return "", f"Timeout após {timeout}s.", 1
    except Exception as e:
        return "", f"Erro: {e}", 1


@tool
def run_pytest(
    path: str = ".",
    test_file: Optional[str] = None,
    verbose: bool = True,
    fail_fast: bool = False,
    repo_path: Optional[str] = None,
) -> str:
    """
    Executa pytest no repositório e retorna o resultado dos testes.

    Args:
        path:       Diretório ou arquivo de testes (padrão: auto-descoberta).
        test_file:  Arquivo específico de testes (ex: "tests/test_auth.py").
        verbose:    Se True, usa -v para output detalhado.
        fail_fast:  Se True, para no primeiro teste que falhar (-x).
        repo_path:  Diretório raiz do repositório.

    Returns:
        Output completo do pytest com resumo de passes/falhas.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    target = test_file or path
    args = ["python", "-m", "pytest", target]

    if verbose:
        args.append("-v")
    if fail_fast:
        args.append("-x")

    # Output colorido desabilitado para parsing
    args += ["--tb=short", "--no-header", "-p", "no:cacheprovider"]

    stdout, stderr, code = _run_shell(args, cwd=cwd)

    output = (stdout + stderr).strip()

    if not output:
        return "[AVISO] pytest não produziu output. Verifique se está instalado: pip install pytest"

    # Extrai linha de resumo
    lines = output.splitlines()
    summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "")

    status = "✅" if code == 0 else "❌"
    header = f"{status} pytest — {summary}" if summary else f"{status} pytest (código: {code})"

    # Limita output longo
    if len(lines) > 100:
        head = "\n".join(lines[:40])
        tail = "\n".join(lines[-30:])
        output = f"{head}\n\n... [{len(lines)-70} linhas omitidas] ...\n\n{tail}"

    return f"{header}\n{'─'*50}\n{output}"


@tool
def run_coverage(
    source: str = ".",
    test_path: str = ".",
    min_coverage: int = 80,
    repo_path: Optional[str] = None,
) -> str:
    """
    Executa pytest com medição de cobertura de código.

    Args:
        source:       Módulo/pacote para medir cobertura (ex: "src", "myapp").
        test_path:    Onde estão os testes.
        min_coverage: Porcentagem mínima esperada (só informativo).
        repo_path:    Diretório raiz do repositório.

    Returns:
        Relatório de cobertura com porcentagem por arquivo.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    args = [
        "python", "-m", "pytest",
        f"--cov={source}",
        "--cov-report=term-missing",
        "--cov-report=json:/tmp/coverage.json",
        "--no-header", "-q",
        test_path,
    ]

    stdout, stderr, code = _run_shell(args, cwd=cwd)
    output = (stdout + stderr).strip()

    if "No module named pytest_cov" in output or "no module named coverage" in output.lower():
        return "[AVISO] pytest-cov não instalado. Execute: pip install pytest-cov"

    lines = output.splitlines()

    # Extrai linha de total
    total_line = next((l for l in lines if "TOTAL" in l), "")
    cov_percent = 0
    if total_line:
        import re
        match = re.search(r'(\d+)%', total_line)
        if match:
            cov_percent = int(match.group(1))

    status = "✅" if cov_percent >= min_coverage else "⚠️"
    header = f"{status} Cobertura: {cov_percent}% (mínimo: {min_coverage}%)"

    return f"{header}\n{'─'*50}\n{output}"


@tool
def run_linter(
    path: str = ".",
    linter: str = "ruff",
    fix: bool = False,
    repo_path: Optional[str] = None,
) -> str:
    """
    Executa linter no código para detectar erros de estilo e problemas.

    Args:
        path:      Arquivo ou diretório para analisar.
        linter:    "ruff" (padrão, mais rápido) ou "pylint".
        fix:       Se True e linter=ruff, aplica correções automáticas.
        repo_path: Diretório raiz do repositório.

    Returns:
        Lista de problemas encontrados pelo linter.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    if linter == "ruff":
        args = ["python", "-m", "ruff", "check", path]
        if fix:
            args.append("--fix")
        args += ["--output-format=concise"]
    elif linter == "pylint":
        args = ["python", "-m", "pylint", path, "--output-format=text", "--score=yes"]
    else:
        return f"[ERRO] Linter desconhecido: '{linter}'. Use 'ruff' ou 'pylint'."

    stdout, stderr, code = _run_shell(args, cwd=cwd)
    output = (stdout + stderr).strip()

    if "No module named ruff" in output or "No module named pylint" in output:
        tool_name = "ruff" if linter == "ruff" else "pylint"
        return f"[AVISO] {tool_name} não instalado. Execute: pip install {tool_name}"

    if not output:
        return f"✅ {linter}: Nenhum problema encontrado em '{path}'"

    lines = output.splitlines()
    error_count   = sum(1 for l in lines if ": E" in l or "error" in l.lower())
    warning_count = sum(1 for l in lines if ": W" in l or "warning" in l.lower())

    status = "❌" if error_count > 0 else "⚠️" if warning_count > 0 else "✅"
    header = f"{status} {linter}: {error_count} erro(s), {warning_count} aviso(s)"

    return f"{header}\n{'─'*50}\n{output}"


@tool
def run_type_check(
    path: str = ".",
    strict: bool = False,
    repo_path: Optional[str] = None,
) -> str:
    """
    Executa mypy para verificação estática de tipos.

    Args:
        path:      Arquivo ou módulo para verificar.
        strict:    Se True, usa --strict mode (mais rigoroso).
        repo_path: Diretório raiz do repositório.

    Returns:
        Lista de erros de tipo encontrados pelo mypy.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    args = ["python", "-m", "mypy", path, "--no-error-summary"]
    if strict:
        args.append("--strict")
    else:
        args += ["--ignore-missing-imports", "--no-strict-optional"]

    stdout, stderr, code = _run_shell(args, cwd=cwd)
    output = (stdout + stderr).strip()

    if "No module named mypy" in output:
        return "[AVISO] mypy não instalado. Execute: pip install mypy"

    if "Success: no issues found" in output or code == 0:
        return f"✅ mypy: Nenhum erro de tipo encontrado em '{path}'"

    lines = output.splitlines()
    error_count = sum(1 for l in lines if ": error:" in l)
    note_count  = sum(1 for l in lines if ": note:" in l)

    header = f"❌ mypy: {error_count} erro(s) de tipo, {note_count} nota(s)"
    return f"{header}\n{'─'*50}\n{output}"


@tool
def run_command(
    command: str,
    repo_path: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    Executa um comando shell arbitrário no repositório.
    Use com cuidado — apenas para comandos de análise/qualidade.
    Nunca use para modificar o sistema ou instalar pacotes sem necessidade.

    Args:
        command:   Comando completo (ex: "python -m bandit -r src/").
        repo_path: Diretório de execução.
        timeout:   Timeout em segundos.

    Returns:
        Output do comando.
    """
    import shlex
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    # Bloqueio básico de segurança
    blocked = ["rm -rf", "sudo", "chmod 777", "> /dev/", "dd if=", "mkfs"]
    if any(b in command for b in blocked):
        return f"[ERRO] Comando bloqueado por segurança: '{command}'"

    try:
        args = shlex.split(command)
    except ValueError as e:
        return f"[ERRO] Comando inválido: {e}"

    stdout, stderr, code = _run_shell(args, cwd=cwd, timeout=timeout)
    output = (stdout + stderr).strip()

    if not output:
        return f"[OK] Comando executado (código: {code}). Sem output."

    status = "✅" if code == 0 else "❌"
    return f"{status} [{code}] {command}\n{'─'*40}\n{output}"


# ─────────────────────────────────────────────────────────────────────────────
# Todas as tools do QA
# ─────────────────────────────────────────────────────────────────────────────

QA_SHELL_TOOLS = [run_pytest, run_coverage, run_linter, run_type_check, run_command]
ALL_QA_TOOLS   = QA_SHELL_TOOLS + QA_TOOLS  # QA_TOOLS = fs tools do filesystem.py

# ─────────────────────────────────────────────────────────────────────────────
# Construção do agente
# ─────────────────────────────────────────────────────────────────────────────

_qa_agent_instance = None

def _get_qa_agent():
    global _qa_agent_instance
    if _qa_agent_instance is None:
        llm = make_llm("qa", temperature=0, max_tokens=4096)
        _qa_agent_instance = create_react_agent(
            model=llm,
            tools=ALL_QA_TOOLS,
            state_modifier=SystemMessage(content=QA_SYSTEM_PROMPT),
        )
    return _qa_agent_instance


# ─────────────────────────────────────────────────────────────────────────────
# Nó do grafo
# ─────────────────────────────────────────────────────────────────────────────

def qa_node(state: AgentState) -> AgentState:
    """
    Nó do QA Agent no grafo LangGraph.

    Recebe a instrução do supervisor, executa a suíte de qualidade
    e retorna um relatório estruturado com status de cada check.
    """
    instruction = state.get("current_instruction", "")
    repo_path   = state.get("repo_path", ".")
    task        = state.get("task", "")

    # Descobre arquivos que foram alterados (contexto para o agente)
    changed_files = state.get("artifacts", {}).get("files_changed", [])
    changed_ctx = ""
    if changed_files:
        changed_ctx = (
            "\n\n## Arquivos recém-modificados (priorize estes):\n"
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

Execute a análise completa de qualidade:
1. Rode pytest para verificar se os testes passam
2. Meça cobertura de código
3. Rode o linter (ruff preferencialmente)
4. Execute type check com mypy

Se algum teste falhar ou a cobertura estiver abaixo de 80%, detalhe
claramente no report para que o Developer possa corrigir.
"""

    try:
        agent  = _get_qa_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        final_message = result["messages"][-1]
        output = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )

        status    = _infer_qa_status(output)
        artifacts = _extract_qa_artifacts(output)

    except Exception as e:
        output    = f"❌ Erro no QA Agent: {type(e).__name__}: {e}"
        status    = "error"
        artifacts = {}

    updates = record_agent_output(
        state=state,
        agent_name="qa",
        output=output,
        status=status,
        artifacts=artifacts,
    )
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_qa_status(output: str) -> str:
    """Infere o status da execução do QA a partir do report."""
    lower = output.lower()

    critical = ["failed", "error", " 0%", "syntax error", "import error"]
    warnings  = ["warning", "aviso", "abaixo", "faltando", "missing"]

    if any(k in lower for k in critical):
        # Falhas em testes são críticas
        if "passed" in lower and "failed" not in lower:
            pass  # tudo passou, pode ser só aviso
        else:
            return "error"

    if any(k in lower for k in warnings):
        return "warning"

    return "success"


def _extract_qa_artifacts(output: str) -> dict:
    """Extrai métricas estruturadas do report do QA."""
    import re
    artifacts = {}

    # Cobertura
    cov_match = re.search(r'cobertura[:\s]+(\d+)%', output, re.IGNORECASE)
    if not cov_match:
        cov_match = re.search(r'(\d+)%\s*(?:coverage|cobertura)', output, re.IGNORECASE)
    if cov_match:
        artifacts["coverage_percent"] = int(cov_match.group(1))

    # Resultado pytest
    pytest_match = re.search(r'(\d+) passed', output)
    if pytest_match:
        artifacts["tests_passed"] = int(pytest_match.group(1))

    failed_match = re.search(r'(\d+) failed', output)
    if failed_match:
        artifacts["tests_failed"] = int(failed_match.group(1))
        artifacts["qa_approved"]  = False
    else:
        artifacts["qa_approved"] = True

    # Erros de linting
    lint_match = re.search(r'(\d+) erro', output, re.IGNORECASE)
    if lint_match:
        artifacts["lint_errors"] = int(lint_match.group(1))

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Exportações
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["qa_node", "QA_SHELL_TOOLS", "ALL_QA_TOOLS"]