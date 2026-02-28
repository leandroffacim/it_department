"""
git_tools.py
─────────────────────────────────────────────────────────────────────────────
Ferramentas de controle de versão (Git) para o IT Department Multi-Agent System.
Usadas principalmente pelo Developer Agent, mas disponíveis para todos.

Ferramentas:
  • git_status       — arquivos modificados, staged, untracked
  • git_diff         — diff do working tree ou entre commits
  • git_log          — histórico de commits
  • git_add          — adiciona arquivos ao staging area
  • git_commit       — cria um commit semântico
  • git_branch       — lista branches ou cria uma nova
  • git_checkout     — troca de branch ou restaura arquivo
  • git_show_commit  — detalhes de um commit específico
  • git_blame        — quem escreveu cada linha de um arquivo
  • git_stash        — salva/restaura mudanças temporárias
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

# Usa o mesmo base path do filesystem.py se disponível
from tools.filesystem import _get_allowed_base_path

# Timeout para comandos git (segundos)
GIT_TIMEOUT = int(os.environ.get("ITDEPT_GIT_TIMEOUT", "30"))

# Máximo de linhas de diff retornadas
MAX_DIFF_LINES = int(os.environ.get("ITDEPT_MAX_DIFF_LINES", "300"))


# ─────────────────────────────────────────────────────────────────────────────
# Helper interno
# ─────────────────────────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: Optional[str] = None) -> tuple[str, str, int]:
    """
    Executa um comando git e retorna (stdout, stderr, returncode).

    Args:
        args: Argumentos do git (sem o 'git' inicial).
        cwd:  Diretório de trabalho. Usa ALLOWED_BASE_PATH se None.

    Returns:
        Tupla (stdout, stderr, returncode).
    """
    workdir = cwd or str(_get_allowed_base_path())

    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "git não encontrado no PATH do sistema.", 127
    except subprocess.TimeoutExpired:
        return "", f"Timeout: comando git demorou mais de {GIT_TIMEOUT}s.", 1
    except Exception as e:
        return "", f"Erro inesperado ao executar git: {e}", 1


def _is_git_repo(path: Optional[str] = None) -> bool:
    """Verifica se o diretório é um repositório git válido."""
    _, _, code = _run_git(["rev-parse", "--git-dir"], cwd=path)
    return code == 0


def _format_result(stdout: str, stderr: str, returncode: int, success_prefix: str = "") -> str:
    """Formata o resultado de um comando git para retorno da tool."""
    if returncode == 0:
        output = stdout.strip()
        if not output:
            return f"[OK] {success_prefix or 'Comando executado com sucesso.'}"
        return f"[OK] {success_prefix}\n{output}" if success_prefix else output
    else:
        error = stderr.strip() or stdout.strip()
        return f"[ERRO] {error}"


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def git_status(repo_path: Optional[str] = None) -> str:
    """
    Mostra o estado atual do repositório: arquivos modificados, staged e untracked.

    Args:
        repo_path: Caminho do repositório (padrão: workspace).

    Returns:
        Status formatado com seções: staged, modificados, não-rastreados.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    # Status legível
    stdout, stderr, code = _run_git(["status", "--short", "--branch"], cwd=cwd)
    if code != 0:
        return _format_result(stdout, stderr, code)

    if not stdout.strip():
        return "[OK] Working tree limpo. Nenhuma mudança pendente."

    lines = stdout.strip().splitlines()
    branch_line = lines[0] if lines[0].startswith("##") else ""
    file_lines  = [l for l in lines if not l.startswith("##")]

    staged    = [l for l in file_lines if l[0] in ("M", "A", "D", "R", "C") and l[0] != " "]
    modified  = [l for l in file_lines if l[1] in ("M", "D") and l[1] != " "]
    untracked = [l for l in file_lines if l.startswith("??")]

    sections = []
    if branch_line:
        branch = branch_line.replace("## ", "").split("...")[0]
        sections.append(f"🌿 Branch: {branch}")

    if staged:
        sections.append("📦 Staged (prontos para commit):")
        sections.extend(f"   {l}" for l in staged)

    if modified:
        sections.append("✏️  Modificados (não staged):")
        sections.extend(f"   {l}" for l in modified)

    if untracked:
        sections.append(f"❓ Não-rastreados: {len(untracked)} arquivo(s)")

    return "\n".join(sections) if sections else stdout.strip()


@tool
def git_diff(
    path: Optional[str] = None,
    staged: bool = False,
    commit: Optional[str] = None,
    repo_path: Optional[str] = None,
) -> str:
    """
    Mostra as diferenças no repositório.

    Args:
        path:      Arquivo específico para ver diff (None = todos).
        staged:    Se True, mostra diff do staging area (--cached).
        commit:    Hash do commit para comparar com HEAD (ex: "abc123").
        repo_path: Caminho do repositório (padrão: workspace).

    Returns:
        Diff formatado, truncado em MAX_DIFF_LINES linhas se necessário.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    args = ["diff"]

    if staged:
        args.append("--cached")

    if commit:
        args.extend([commit, "HEAD"])

    args += ["--stat", "--", path] if path else ["--stat"]
    stat_out, _, stat_code = _run_git(args, cwd=cwd)

    # Diff completo (sem --stat)
    full_args = ["diff"]
    if staged:
        full_args.append("--cached")
    if commit:
        full_args.extend([commit, "HEAD"])
    if path:
        full_args += ["--", path]

    stdout, stderr, code = _run_git(full_args, cwd=cwd)

    if code != 0:
        return _format_result(stdout, stderr, code)

    if not stdout.strip():
        label = "staging area" if staged else "working tree"
        return f"[INFO] Sem diferenças no {label}."

    lines = stdout.splitlines()
    truncated = False
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES]
        truncated = True

    output = "\n".join(lines)
    if truncated:
        output += f"\n\n... (truncado em {MAX_DIFF_LINES} linhas. Use path= para focar em um arquivo)"

    if stat_out.strip():
        output = f"📊 Resumo:\n{stat_out.strip()}\n\n{'─'*40}\n" + output

    return output


@tool
def git_log(
    max_commits: int = 10,
    path: Optional[str] = None,
    oneline: bool = True,
    repo_path: Optional[str] = None,
) -> str:
    """
    Mostra o histórico de commits do repositório.

    Args:
        max_commits: Número máximo de commits (padrão: 10).
        path:        Filtrar commits que tocaram em um arquivo específico.
        oneline:     Se True, formato compacto (hash + título).
        repo_path:   Caminho do repositório.

    Returns:
        Lista de commits formatada.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    args = ["log", f"-{max_commits}"]

    if oneline:
        args += ["--pretty=format:%C(yellow)%h%Creset %C(cyan)%ar%Creset %s %C(dim)[%an]%Creset"]
    else:
        args += ["--pretty=format:%H%n%an <%ae>%n%ai%n%s%n%b%n---"]

    if path:
        args += ["--", path]

    stdout, stderr, code = _run_git(args, cwd=cwd)

    if code != 0:
        return _format_result(stdout, stderr, code)

    if not stdout.strip():
        return "[INFO] Nenhum commit encontrado."

    header = f"📜 Últimos {max_commits} commits"
    if path:
        header += f" em '{path}'"
    return header + "\n" + stdout.strip()


@tool
def git_add(
    paths: list[str],
    repo_path: Optional[str] = None,
) -> str:
    """
    Adiciona arquivos ao staging area para o próximo commit.

    Args:
        paths:     Lista de caminhos para adicionar. Use ["."] para todos.
        repo_path: Caminho do repositório.

    Returns:
        Confirmação com lista de arquivos staged.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    if not paths:
        return "[ERRO] Forneça ao menos um caminho. Use ['.'] para todos os arquivos."

    stdout, stderr, code = _run_git(["add"] + paths, cwd=cwd)

    if code != 0:
        return _format_result(stdout, stderr, code)

    # Confirma o que foi staged
    staged_out, _, _ = _run_git(["diff", "--cached", "--name-only"], cwd=cwd)
    staged_files = staged_out.strip().splitlines()

    if not staged_files:
        return "[AVISO] git add executado mas nenhum arquivo foi staged (já estavam atualizados?)."

    files_list = "\n".join(f"   + {f}" for f in staged_files)
    return f"[OK] {len(staged_files)} arquivo(s) adicionado(s) ao staging:\n{files_list}"


@tool
def git_commit(
    message: str,
    repo_path: Optional[str] = None,
) -> str:
    """
    Cria um commit com a mensagem fornecida.
    Use o formato semântico: tipo(escopo): descrição
    Tipos válidos: feat, fix, refactor, chore, test, docs, style, perf

    Args:
        message:   Mensagem do commit (ex: "feat(auth): add JWT validation").
        repo_path: Caminho do repositório.

    Returns:
        Confirmação com hash do commit criado.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    if not message.strip():
        return "[ERRO] Mensagem de commit não pode ser vazia."

    # Verifica se há algo staged
    staged_out, _, _ = _run_git(["diff", "--cached", "--name-only"], cwd=cwd)
    if not staged_out.strip():
        return (
            "[ERRO] Staging area vazio. Use git_add() antes de commitar.\n"
            "Dica: git_add(paths=['.']) para adicionar todas as mudanças."
        )

    # Valida formato semântico (soft warning, não bloqueia)
    import re
    semantic_pattern = r'^(feat|fix|refactor|chore|test|docs|style|perf|ci|build)(\(.+\))?: .+'
    is_semantic = bool(re.match(semantic_pattern, message.strip()))

    stdout, stderr, code = _run_git(["commit", "-m", message], cwd=cwd)

    if code != 0:
        return _format_result(stdout, stderr, code)

    # Extrai hash do commit
    hash_out, _, _ = _run_git(["rev-parse", "--short", "HEAD"], cwd=cwd)
    commit_hash = hash_out.strip()

    result = f"[OK] Commit criado: {commit_hash}\n📝 {message}"
    if not is_semantic:
        result += (
            "\n\n⚠️  Sugestão: use formato semântico "
            "(ex: 'feat(escopo): descrição')"
        )
    return result


@tool
def git_branch(
    name: Optional[str] = None,
    create: bool = False,
    repo_path: Optional[str] = None,
) -> str:
    """
    Lista branches existentes ou cria uma nova branch.

    Args:
        name:      Nome da branch para criar (obrigatório se create=True).
        create:    Se True, cria a branch e faz checkout nela.
        repo_path: Caminho do repositório.

    Returns:
        Lista de branches ou confirmação de criação.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    if create:
        if not name:
            return "[ERRO] Forneça o nome da branch com name='nome-da-branch'."
        stdout, stderr, code = _run_git(["checkout", "-b", name], cwd=cwd)
        return _format_result(stdout, stderr, code, f"Branch '{name}' criada e ativa.")

    # Lista branches
    stdout, stderr, code = _run_git(["branch", "-a", "--color=never"], cwd=cwd)
    if code != 0:
        return _format_result(stdout, stderr, code)

    lines = stdout.strip().splitlines()
    current = next((l.replace("* ", "").strip() for l in lines if l.startswith("* ")), "?")
    all_branches = [l.strip().replace("* ", "") for l in lines]

    local  = [b for b in all_branches if not b.startswith("remotes/")]
    remote = [b for b in all_branches if b.startswith("remotes/")]

    output = [f"🌿 Branch atual: {current}", f"\nLocais ({len(local)}):"]
    output += [f"   {'→' if b == current else ' '} {b}" for b in local]

    if remote:
        output += [f"\nRemotas ({len(remote)}):"]
        output += [f"   {b}" for b in remote[:10]]
        if len(remote) > 10:
            output.append(f"   ... e mais {len(remote)-10}")

    return "\n".join(output)


@tool
def git_checkout(
    target: str,
    repo_path: Optional[str] = None,
) -> str:
    """
    Troca para uma branch existente ou restaura um arquivo para o estado do HEAD.

    Args:
        target:    Nome da branch OU caminho de arquivo para restaurar.
        repo_path: Caminho do repositório.

    Returns:
        Confirmação da operação.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    # Detecta se é arquivo ou branch
    target_path = Path(cwd) / target
    if target_path.exists() and target_path.is_file():
        stdout, stderr, code = _run_git(["checkout", "--", target], cwd=cwd)
        return _format_result(stdout, stderr, code, f"Arquivo '{target}' restaurado para HEAD.")
    else:
        stdout, stderr, code = _run_git(["checkout", target], cwd=cwd)
        return _format_result(stdout, stderr, code, f"Switched para branch '{target}'.")


@tool
def git_show_commit(
    commit: str = "HEAD",
    repo_path: Optional[str] = None,
) -> str:
    """
    Mostra os detalhes e o diff de um commit específico.

    Args:
        commit:    Hash ou referência do commit (padrão: HEAD).
        repo_path: Caminho do repositório.

    Returns:
        Detalhes do commit: autor, data, mensagem e diff.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    stdout, stderr, code = _run_git(
        ["show", "--stat", commit,
         "--pretty=format:🔖 %h  %s%n👤 %an <%ae>%n📅 %ai%n"],
        cwd=cwd,
    )
    if code != 0:
        return _format_result(stdout, stderr, code)

    lines = stdout.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES]
        lines.append(f"... (truncado em {MAX_DIFF_LINES} linhas)")

    return "\n".join(lines)


@tool
def git_blame(
    path: str,
    start_line: int = 1,
    end_line: int = 50,
    repo_path: Optional[str] = None,
) -> str:
    """
    Mostra quem escreveu cada linha de um arquivo (git blame).
    Útil para o Reviewer entender a autoria e contexto histórico.

    Args:
        path:       Caminho do arquivo relativo ao repositório.
        start_line: Linha inicial (padrão: 1).
        end_line:   Linha final (padrão: 50).
        repo_path:  Caminho do repositório.

    Returns:
        Blame formatado com hash, autor, data e conteúdo da linha.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    stdout, stderr, code = _run_git(
        ["blame", f"-L{start_line},{end_line}", "--date=short", path],
        cwd=cwd,
    )
    return _format_result(stdout, stderr, code)


@tool
def git_stash(
    action: str = "push",
    message: Optional[str] = None,
    repo_path: Optional[str] = None,
) -> str:
    """
    Salva ou restaura mudanças temporárias com git stash.

    Args:
        action:    "push" (salvar), "pop" (restaurar último), "list" (listar).
        message:   Descrição do stash (só para action="push").
        repo_path: Caminho do repositório.

    Returns:
        Confirmação da operação.
    """
    cwd = repo_path or str(_get_allowed_base_path())

    if not _is_git_repo(cwd):
        return f"[ERRO] '{cwd}' não é um repositório git."

    if action == "push":
        args = ["stash", "push"]
        if message:
            args += ["-m", message]
        stdout, stderr, code = _run_git(args, cwd=cwd)
        return _format_result(stdout, stderr, code, "Mudanças salvas no stash.")

    elif action == "pop":
        stdout, stderr, code = _run_git(["stash", "pop"], cwd=cwd)
        return _format_result(stdout, stderr, code, "Último stash restaurado.")

    elif action == "list":
        stdout, stderr, code = _run_git(["stash", "list"], cwd=cwd)
        if not stdout.strip():
            return "[INFO] Nenhum stash salvo."
        return stdout.strip()

    else:
        return f"[ERRO] action inválido: '{action}'. Use 'push', 'pop' ou 'list'."


# ─────────────────────────────────────────────────────────────────────────────
# Exportações para uso nos agentes
# ─────────────────────────────────────────────────────────────────────────────

GIT_TOOLS = [
    git_status,
    git_diff,
    git_log,
    git_add,
    git_commit,
    git_branch,
    git_checkout,
    git_show_commit,
    git_blame,
    git_stash,
]

# Subconjuntos por agente
DEVELOPER_GIT_TOOLS = [git_status, git_diff, git_add, git_commit, git_stash]
REVIEWER_GIT_TOOLS  = [git_log, git_diff, git_blame, git_show_commit, git_status]
QA_GIT_TOOLS        = [git_status, git_diff, git_log]
DEVOPS_GIT_TOOLS    = [git_branch, git_checkout, git_status, git_log, git_stash]


# ─────────────────────────────────────────────────────────────────────────────
# Teste rápido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(git_status.invoke({}))
    print()
    print(git_log.invoke({"max_commits": 5}))
    print()
    print(git_branch.invoke({}))
