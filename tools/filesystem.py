"""
filesystem.py
─────────────────────────────────────────────────────────────────────────────
Ferramentas de sistema de arquivos para o IT Department Multi-Agent System.
Todos os agentes usam estas tools via @tool do LangChain.

Ferramentas disponíveis:
  • read_file          — lê o conteúdo de um arquivo
  • write_file         — cria ou sobrescreve um arquivo
  • append_file        — acrescenta conteúdo ao final de um arquivo
  • delete_file        — remove um arquivo
  • list_directory     — lista arquivos e pastas de um diretório
  • search_in_files    — busca texto/regex em arquivos do repositório
  • get_file_info      — metadata de um arquivo (tamanho, datas, linhas)
  • create_directory   — cria um diretório (com parents)
  • move_file          — move ou renomeia um arquivo
  • copy_file          — copia um arquivo para outro destino
  • get_repo_tree      — árvore completa do repositório (respeita .gitignore)
  • patch_file         — aplica um diff/patch em um arquivo existente

Segurança:
  • Todos os caminhos são validados contra ALLOWED_BASE_PATH para evitar
    path traversal (../../etc/passwd, etc.).
  • Extensões bloqueadas por padrão (configurável via BLOCKED_EXTENSIONS).
  • Tamanho máximo de leitura configurável (MAX_READ_BYTES).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import difflib
import fnmatch
import mimetypes
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Configuração global
# ─────────────────────────────────────────────────────────────────────────────

# Caminho base permitido — protege contra path traversal.
# Ordem de precedência:
#   1.set_base_path(path) chamado em runtime pelo graph.py
#   2.Variável de ambiente ITDEPT_BASE_PATH
#   3. Diretório atual (cwd) como fallback
ALLOWED_BASE_PATH: Path = Path(
    os.environ.get("ITDEPT_BASE_PATH", str(Path.cwd()))
).resolve()


def set_base_path(path: str) -> None:
    """
    Atualiza ALLOWED_BASE_PATH em runtime.
    Deve ser chamado antes de qualquer tool, passando o repo_path do AgentState.
    Chamado automaticamente pelo graph.py ao iniciar cada execução.
    """
    global ALLOWED_BASE_PATH
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise ValueError(f"Repositorio nao encontrado: '{path}'")
    if not resolved.is_dir():
        raise ValueError(f"'{path}' nao e um diretorio.")
    ALLOWED_BASE_PATH = resolved

# Tamanho máximo de leitura por arquivo (padrão: 1 MB)
MAX_READ_BYTES: int = int(os.environ.get("ITDEPT_MAX_READ_BYTES", 1_048_576))

# Extensões que nunca serão lidas ou escritas pelos agentes
BLOCKED_EXTENSIONS: set[str] = {
    ".exe", ".dll", ".so", ".dylib",
    ".bin", ".iso", ".img",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pdf", ".xls", ".xlsx", ".doc", ".docx",
    ".pyc", ".pyo",
}

# Pastas ignoradas ao listar/buscar
IGNORED_DIRS: set[str] = {
    ".git", "__pycache__", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", "node_modules", ".venv", "venv",
    "dist", "build", ".tox", ".eggs",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _safe_path(raw: str) -> Path:
    """
    Resolve o caminho e garante que está dentro de ALLOWED_BASE_PATH.
    Levanta ValueError em caso de path traversal.
    """
    path = (ALLOWED_BASE_PATH / raw).resolve()
    if not str(path).startswith(str(ALLOWED_BASE_PATH)):
        raise ValueError(
            f"Acesso negado: '{raw}' está fora do workspace permitido "
            f"({ALLOWED_BASE_PATH})."
        )
    return path


def _check_extension(path: Path, *, writing: bool = False) -> None:
    """Levanta ValueError se a extensão for bloqueada."""
    ext = path.suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        action = "escrever em" if writing else "ler"
        raise ValueError(f"Extensão bloqueada: não é permitido {action} '{ext}'.")


def _read_gitignore(base: Path) -> list[str]:
    """Retorna os padrões do .gitignore raiz (se existir)."""
    gi = base / ".gitignore"
    if gi.is_file():
        lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    return []


def _is_ignored_by_gitignore(path: Path, patterns: list[str], base: Path) -> bool:
    rel = str(path.relative_to(base))
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    Lê e retorna o conteúdo de um arquivo de texto.

    Args:
        path:     Caminho relativo ao workspace (ex: "src/main.py").
        encoding: Encoding do arquivo (padrão: utf-8).

    Returns:
        Conteúdo completo do arquivo como string, com cabeçalho informativo.
    """
    try:
        target = _safe_path(path)
        _check_extension(target)

        if not target.exists():
            return f"[ERRO] Arquivo não encontrado: '{path}'"
        if not target.is_file():
            return f"[ERRO] '{path}' não é um arquivo."

        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return (
                f"[ERRO] Arquivo muito grande: {size:,} bytes "
                f"(máximo permitido: {MAX_READ_BYTES:,} bytes). "
                f"Use search_in_files para localizar trechos específicos."
            )

        content = target.read_text(encoding=encoding, errors="replace")
        lines = content.splitlines()
        header = (
            f"# ── {path} ──────────────────────────────\n"
            f"# linhas: {len(lines)} | tamanho: {size:,} bytes | encoding: {encoding}\n"
            f"# ─────────────────────────────────────────\n"
        )
        return header + content

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao ler '{path}': {e}"


@tool
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Cria ou sobrescreve um arquivo com o conteúdo fornecido.
    Cria os diretórios intermediários automaticamente.

    Args:
        path:     Caminho relativo ao workspace.
        content:  Conteúdo a ser escrito.
        encoding: Encoding (padrão: utf-8).

    Returns:
        Mensagem de sucesso com informações do arquivo escrito.
    """
    try:
        target = _safe_path(path)
        _check_extension(target, writing=True)

        target.parent.mkdir(parents=True, exist_ok=True)

        existed = target.exists()
        target.write_text(content, encoding=encoding)

        lines = content.splitlines()
        action = "Atualizado" if existed else "Criado"
        return (
            f"[OK] {action}: '{path}' | "
            f"{len(lines)} linhas | {len(content.encode(encoding)):,} bytes"
        )

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao escrever '{path}': {e}"


@tool
def append_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Acrescenta conteúdo ao final de um arquivo (cria se não existir).

    Args:
        path:    Caminho relativo ao workspace.
        content: Texto a ser adicionado no final.

    Returns:
        Mensagem de sucesso com o novo tamanho do arquivo.
    """
    try:
        target = _safe_path(path)
        _check_extension(target, writing=True)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("a", encoding=encoding) as f:
            f.write(content)

        new_size = target.stat().st_size
        return f"[OK] Conteúdo adicionado em '{path}' | tamanho total: {new_size:,} bytes"

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao fazer append em '{path}': {e}"


@tool
def delete_file(path: str) -> str:
    """
    Remove um arquivo do sistema de arquivos.

    Args:
        path: Caminho relativo ao workspace.

    Returns:
        Confirmação de remoção.
    """
    try:
        target = _safe_path(path)

        if not target.exists():
            return f"[AVISO] Arquivo não encontrado (já removido?): '{path}'"
        if target.is_dir():
            return f"[ERRO] '{path}' é um diretório. Use delete_directory (não implementado por segurança)."

        target.unlink()
        return f"[OK] Arquivo removido: '{path}'"

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao remover '{path}': {e}"


@tool
def list_directory(path: str = ".", recursive: bool = False) -> str:
    """
    Lista o conteúdo de um diretório.

    Args:
        path:      Caminho relativo ao workspace (padrão: raiz do workspace).
        recursive: Se True, lista recursivamente (ignorando pastas de build/venv).

    Returns:
        Lista formatada de arquivos e diretórios.
    """
    try:
        target = _safe_path(path)

        if not target.exists():
            return f"[ERRO] Caminho não encontrado: '{path}'"
        if not target.is_dir():
            return f"[ERRO] '{path}' não é um diretório."

        entries: list[str] = []

        if recursive:
            for item in sorted(target.rglob("*")):
                if any(part in IGNORED_DIRS for part in item.parts):
                    continue
                rel = item.relative_to(target)
                indent = "  " * (len(rel.parts) - 1)
                icon = "📁" if item.is_dir() else "📄"
                size_info = ""
                if item.is_file():
                    try:
                        size_info = f"  ({item.stat().st_size:,}b)"
                    except OSError:
                        pass
                entries.append(f"{indent}{icon} {item.name}{size_info}")
        else:
            for item in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name)):
                icon = "📁" if item.is_dir() else "📄"
                size_info = ""
                if item.is_file():
                    try:
                        size_info = f"  ({item.stat().st_size:,}b)"
                    except OSError:
                        pass
                entries.append(f"{icon} {item.name}{size_info}")

        if not entries:
            return f"[INFO] Diretório vazio: '{path}'"

        header = f"📂 {path}/ — {len(entries)} itens\n" + "─" * 40
        return header + "\n" + "\n".join(entries)

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao listar '{path}': {e}"


@tool
def search_in_files(
    query: str,
    path: str = ".",
    file_pattern: str = "*.py",
    use_regex: bool = False,
    case_sensitive: bool = False,
    context_lines: int = 2,
    max_results: int = 50,
) -> str:
    """
    Busca texto ou padrão regex em arquivos do repositório.

    Args:
        query:          Texto ou expressão regular a buscar.
        path:           Diretório raiz da busca (padrão: workspace inteiro).
        file_pattern:   Padrão glob para filtrar arquivos (ex: "*.py", "*.js").
        use_regex:      Se True, trata query como regex.
        case_sensitive: Se True, busca com distinção de maiúsculas.
        context_lines:  Linhas de contexto antes/depois de cada match.
        max_results:    Limite de resultados retornados.

    Returns:
        Resultados formatados com nome do arquivo, número da linha e contexto.
    """
    try:
        base = _safe_path(path)
        gi_patterns = _read_gitignore(ALLOWED_BASE_PATH)
        flags = 0 if case_sensitive else re.IGNORECASE

        if use_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error as e:
                return f"[ERRO] Regex inválida: {e}"
        else:
            escaped = re.escape(query)
            pattern = re.compile(escaped, flags)

        results: list[str] = []
        files_searched = 0
        total_matches = 0

        for filepath in sorted(base.rglob(file_pattern)):
            if any(part in IGNORED_DIRS for part in filepath.parts):
                continue
            if _is_ignored_by_gitignore(filepath, gi_patterns, ALLOWED_BASE_PATH):
                continue
            if not filepath.is_file():
                continue
            if filepath.stat().st_size > MAX_READ_BYTES:
                continue

            try:
                lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue

            files_searched += 1
            file_hits: list[str] = []

            for i, line in enumerate(lines):
                if pattern.search(line):
                    total_matches += 1
                    if total_matches > max_results:
                        break

                    start = max(0, i - context_lines)
                    end   = min(len(lines), i + context_lines + 1)
                    ctx_lines = []
                    for j in range(start, end):
                        prefix = ">>> " if j == i else "    "
                        ctx_lines.append(f"  {prefix}{j+1:4d} │ {lines[j]}")

                    file_hits.append("\n".join(ctx_lines))

            if file_hits:
                rel = filepath.relative_to(ALLOWED_BASE_PATH)
                results.append(
                    f"\n📄 {rel} ({len(file_hits)} ocorrência(s))\n"
                    + "\n  ···\n".join(file_hits)
                )

            if total_matches > max_results:
                break

        if not results:
            return (
                f"[INFO] Nenhum resultado para '{query}' "
                f"em '{path}/**/{file_pattern}' "
                f"({files_searched} arquivo(s) verificado(s))."
            )

        header = (
            f"🔍 Busca: '{query}' | padrão: {file_pattern}\n"
            f"   {total_matches} match(es) em {len(results)} arquivo(s) "
            f"| {files_searched} arquivo(s) verificado(s)"
        )
        if total_matches > max_results:
            header += f"\n   ⚠️  Limite de {max_results} resultados atingido."

        return header + "\n" + "═" * 50 + "".join(results)

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha na busca: {e}"


@tool
def get_file_info(path: str) -> str:
    """
    Retorna metadata detalhada de um arquivo ou diretório.

    Args:
        path: Caminho relativo ao workspace.

    Returns:
        Informações: tamanho, datas, permissões, tipo MIME, total de linhas.
    """
    try:
        target = _safe_path(path)

        if not target.exists():
            return f"[ERRO] Caminho não encontrado: '{path}'"

        stat = target.stat()
        mime, _ = mimetypes.guess_type(str(target))
        is_file = target.is_file()

        info_lines = [
            f"📄 {path}",
            f"   tipo:      {'arquivo' if is_file else 'diretório'}",
            f"   tamanho:   {stat.st_size:,} bytes",
            f"   criado:    {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
            f"   modificado:{datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if is_file:
            info_lines.append(f"   mime:      {mime or 'desconhecido'}")
            info_lines.append(f"   extensão:  {target.suffix or '(sem extensão)'}")
            if mime and mime.startswith("text"):
                try:
                    content = target.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()
                    non_empty = [l for l in lines if l.strip()]
                    info_lines.append(f"   linhas:    {len(lines)} total / {len(non_empty)} não-vazias")
                except OSError:
                    pass
        else:
            children = list(target.iterdir())
            info_lines.append(f"   itens:     {len(children)}")

        return "\n".join(info_lines)

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao obter info de '{path}': {e}"


@tool
def create_directory(path: str) -> str:
    """
    Cria um diretório (e todos os pais necessários).

    Args:
        path: Caminho relativo ao workspace.

    Returns:
        Confirmação de criação.
    """
    try:
        target = _safe_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return f"[OK] Diretório criado (ou já existia): '{path}'"
    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao criar diretório '{path}': {e}"


@tool
def move_file(source: str, destination: str) -> str:
    """
    Move ou renomeia um arquivo dentro do workspace.

    Args:
        source:      Caminho relativo de origem.
        destination: Caminho relativo de destino.

    Returns:
        Confirmação da operação.
    """
    try:
        src = _safe_path(source)
        dst = _safe_path(destination)

        if not src.exists():
            return f"[ERRO] Origem não encontrada: '{source}'"

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"[OK] Movido: '{source}' → '{destination}'"

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao mover '{source}': {e}"


@tool
def copy_file(source: str, destination: str) -> str:
    """
    Copia um arquivo para outro destino dentro do workspace.

    Args:
        source:      Caminho relativo de origem.
        destination: Caminho relativo de destino.

    Returns:
        Confirmação da operação.
    """
    try:
        src = _safe_path(source)
        dst = _safe_path(destination)
        _check_extension(src)
        _check_extension(dst, writing=True)

        if not src.is_file():
            return f"[ERRO] Origem não é um arquivo: '{source}'"

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return f"[OK] Copiado: '{source}' → '{destination}'"

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao copiar '{source}': {e}"


@tool
def get_repo_tree(path: str = ".", max_depth: int = 4) -> str:
    """
    Gera uma árvore visual do repositório, respeitando .gitignore e pastas ignoradas.

    Args:
        path:      Raiz da árvore (padrão: workspace).
        max_depth: Profundidade máxima de recursão (padrão: 4).

    Returns:
        Árvore formatada em estilo `tree` do Unix.
    """
    try:
        base = _safe_path(path)
        gi_patterns = _read_gitignore(ALLOWED_BASE_PATH)

        if not base.is_dir():
            return f"[ERRO] '{path}' não é um diretório."

        lines: list[str] = [f"📂 {base.name}/"]

        def _walk(directory: Path, prefix: str, depth: int) -> None:
            if depth > max_depth:
                lines.append(f"{prefix}└── ...")
                return

            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower()),
                )
            except PermissionError:
                return

            children = [
                c for c in children
                if c.name not in IGNORED_DIRS
                and not _is_ignored_by_gitignore(c, gi_patterns, ALLOWED_BASE_PATH)
            ]

            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "

                if child.is_dir():
                    lines.append(f"{prefix}{connector}📁 {child.name}/")
                    _walk(child, prefix + extension, depth + 1)
                else:
                    size = ""
                    try:
                        size = f"  ({child.stat().st_size:,}b)"
                    except OSError:
                        pass
                    lines.append(f"{prefix}{connector}📄 {child.name}{size}")

        _walk(base, "", 1)
        return "\n".join(lines)

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao gerar árvore de '{path}': {e}"


@tool
def patch_file(path: str, original_snippet: str, new_snippet: str) -> str:
    """
    Substitui uma ocorrência exata de texto em um arquivo (patch cirúrgico).
    Ideal para o Developer Agent fazer edições pontuais sem reescrever o arquivo inteiro.

    Args:
        path:             Caminho relativo ao workspace.
        original_snippet: Trecho EXATO que será substituído.
        new_snippet:      Novo conteúdo que substituirá o trecho.

    Returns:
        Confirmação com diff resumido da mudança.
    """
    try:
        target = _safe_path(path)
        _check_extension(target, writing=True)

        if not target.is_file():
            return f"[ERRO] Arquivo não encontrado: '{path}'"

        original_content = target.read_text(encoding="utf-8", errors="replace")

        if original_snippet not in original_content:
            # tenta dar dica de qual trecho está parecido
            ratio_best = 0.0
            for i in range(0, len(original_content) - len(original_snippet), 50):
                chunk = original_content[i : i + len(original_snippet)]
                r = difflib.SequenceMatcher(None, original_snippet, chunk).ratio()
                if r > ratio_best:
                    ratio_best = r

            return (
                f"[ERRO] Trecho original não encontrado em '{path}'.\n"
                f"       Similaridade máxima encontrada: {ratio_best:.0%}.\n"
                f"       Use read_file para verificar o conteúdo atual e ajustar o snippet."
            )

        count = original_content.count(original_snippet)
        if count > 1:
            return (
                f"[ERRO] O trecho aparece {count}x no arquivo '{path}'. "
                f"Forneça um snippet mais específico para identificar a ocorrência correta."
            )

        new_content = original_content.replace(original_snippet, new_snippet, 1)
        target.write_text(new_content, encoding="utf-8")

        # gera diff compacto
        diff = difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        )
        diff_str = "".join(list(diff)[:40])  # máximo 40 linhas no output

        return (
            f"[OK] Patch aplicado em '{path}'\n"
            f"─── diff ───────────────────────────────────\n"
            f"{diff_str}"
            f"────────────────────────────────────────────"
        )

    except ValueError as e:
        return f"[ERRO] {e}"
    except Exception as e:
        return f"[ERRO] Falha ao aplicar patch em '{path}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Exportações para uso no grafo LangGraph
# ─────────────────────────────────────────────────────────────────────────────

ALL_FILESYSTEM_TOOLS = [
    read_file,
    write_file,
    append_file,
    delete_file,
    list_directory,
    search_in_files,
    get_file_info,
    create_directory,
    move_file,
    copy_file,
    get_repo_tree,
    patch_file,
]

# Subconjuntos por agente — importe direto nos nós do grafo
DEVELOPER_TOOLS  = [read_file, write_file, patch_file, list_directory, get_repo_tree]
QA_TOOLS         = [read_file, search_in_files, get_file_info, list_directory]
REVIEWER_TOOLS   = [read_file, search_in_files, get_file_info, get_repo_tree]
DEVOPS_TOOLS     = [read_file, write_file, patch_file, create_directory, copy_file, move_file, list_directory]
DOCS_TOOLS       = [read_file, write_file, append_file, get_repo_tree, search_in_files]


# ─────────────────────────────────────────────────────────────────────────────
# Exemplo de uso direto (python filesystem.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"Workspace: {ALLOWED_BASE_PATH}\n")
    print("─" * 50)

    # Árvore do workspace atual
    print(get_repo_tree.invoke({"path": ".", "max_depth": 2}))
    print()

    # Busca por TODOs em arquivos Python
    if "--search" in sys.argv:
        print(search_in_files.invoke({
            "query": "TODO",
            "path": ".",
            "file_pattern": "*.py",
        }))