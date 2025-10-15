#!/usr/bin/env python3
"""
md2html_clipboard.py

Convierte Markdown -> HTML y copia HTML al portapapeles como tipo text/html
soportando macOS, Linux (wl-copy/xclip) y Windows (pywin32).

Uso:
    python md2html_clipboard.py input.md
    cat input.md | python md2html_clipboard.py -
"""

from __future__ import annotations
import sys
import os
import shutil
import argparse
import subprocess
import tempfile
import platform
from typing import Optional

# ---------- HTML generation ----------
def md_to_html_with_pandoc(md: str) -> str:
    """Intentar usar pandoc (más fiel). Lanza CalledProcessError si falla."""
    p = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html"],
        input=md.encode("utf8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return p.stdout.decode("utf8")

def md_to_html_with_python_md(md: str) -> str:
    """Fallback: usa python-markdown (simple)."""
    try:
        import markdown
    except Exception as e:
        raise RuntimeError("markdown library not available (pip install markdown)") from e
    # Extensiones comunes
    html = markdown.markdown(md, extensions=["fenced_code", "codehilite", "tables", "toc"])
    # Devolver fragmento HTML (no <html><body> wrapper)
    return html

def markdown_to_html(md: str) -> str:
    """Convierte Markdown a HTML: pandoc si existe, sino python-markdown."""
    # Primero pandoc si está instalado
    if shutil.which("pandoc"):
        try:
            return md_to_html_with_pandoc(md)
        except subprocess.CalledProcessError:
            # fallback
            pass
    # fallback a python markdown
    return md_to_html_with_python_md(md)

# ---------- Clipboard helpers ----------
def copy_html_mac_pbcopy_swift(html: str, swift_exec: Optional[str] = None) -> bool:
    """
    Llama a pbcopy-html.swift o un ejecutable que acepte HTML en stdin y marque tipo html.
    swift_exec: comando (path) al ejecutable; si None busca 'pbcopy-html.swift' en PATH o en cwd.
    """
    if swift_exec is None:
        names = ["pbcopy-html.swift", "./pbcopy-html.swift", "/usr/local/bin/pbcopy-html.swift"]
        swift_exec = None
        for n in names:
            if os.path.isfile(n) and os.access(n, os.X_OK):
                swift_exec = n
                break
        if swift_exec is None:
            swift_exec = shutil.which("pbcopy-html.swift")

    if swift_exec:
        try:
            p = subprocess.run([swift_exec], input=html.encode("utf8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except Exception as e:
            # fall through to other mac method
            print(f"[debug] pbcopy-html call failed: {e}", file=sys.stderr)
            return False
    return False

def copy_html_mac_pyobjc(html: str) -> bool:
    """
    Intentar usar pyobjc/AppKit para setear NSPasteboard HTML.
    Requiere 'pyobjc' instalado (pip install pyobjc).
    """
    try:
        # Import dinámico para no requerir pyobjc si no está
        from AppKit import NSPasteboard, NSPasteboardTypeHTML
        # General pasteboard getter - PyObjC puede tener nombres distintos según versión
        try:
            pb = NSPasteboard.generalPasteboard()
        except AttributeError:
            pb = NSPasteboard.general()
        pb.clearContents()
        html_bytes = html.encode("utf8")
        # setData: PyObjC puede ofrecer setData_forType_ o setData_forType
        try:
            pb.setData_forType_(html_bytes, NSPasteboardTypeHTML)
        except Exception:
            # Fallback: usar 'public.html' label
            try:
                pb.setData_forType_(html_bytes, "public.html")
            except Exception as e:
                print(f"[debug] pb.setData failed: {e}", file=sys.stderr)
                return False
        return True
    except Exception as e:
        # PyObjC no disponible o fallo
        print(f"[debug] pyobjc method failed or not available: {e}", file=sys.stderr)
        return False

def copy_html_linux(html: str) -> bool:
    """
    Usa wl-copy (Wayland) o xclip (X11) si están disponibles.
    wl-copy: wl-copy --type text/html
    xclip: xclip -selection clipboard -t text/html
    """
    # wl-copy
    wl = shutil.which("wl-copy")
    if wl:
        try:
            p = subprocess.run([wl, "--type", "text/html"], input=html.encode("utf8"), check=True)
            return True
        except Exception as e:
            print(f"[debug] wl-copy failed: {e}", file=sys.stderr)
    # xclip
    xclip = shutil.which("xclip")
    if xclip:
        try:
            p = subprocess.run([xclip, "-selection", "clipboard", "-t", "text/html"], input=html.encode("utf8"), check=True)
            return True
        except Exception as e:
            print(f"[debug] xclip failed: {e}", file=sys.stderr)
    # xsel maybe (less robust for html)
    xsel = shutil.which("xsel")
    if xsel:
        try:
            # xsel doesn't set type easily; try as a last resort for plain html
            p = subprocess.run([xsel, "--clipboard", "--input"], input=html.encode("utf8"), check=True)
            print("[warning] xsel used as fallback: may not set MIME type (text/html) correctly.", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[debug] xsel failed: {e}", file=sys.stderr)
    return False

def build_cf_html_fragment(html: str) -> bytes:
    """
    Construye el formato CF_HTML requerido por el portapapeles de Windows.
    Basado en la especificación HTML Clipboard Format.
    Devuelve bytes (UTF-8).
    """
    # We will create a minimal HTML clipboard with StartHTML/EndHTML offsets as 10-digit zero-padded ints.
    # Use HTML fragment including <html><body> wrapper.
    if not html.strip().lower().startswith("<html"):
        fragment = "<html><body>\n" + html + "\n</body></html>"
    else:
        fragment = html
    utf8 = fragment.encode("utf-8")
    # Build header
    start_html =  len("Version:0.9\r\nStartHTML:0000000000\r\nEndHTML:0000000000\r\nStartFragment:0000000000\r\nEndFragment:0000000000\r\n")
    # We'll place fragment markers around the body content
    # Find insertion point for fragment (after <body>)
    # Simple approach: locate <body> tag
    lower = fragment.lower()
    idx_body = lower.find("<body")
    if idx_body != -1:
        idx_body_end = lower.find(">", idx_body)
        if idx_body_end != -1:
            frag_start_pos = idx_body_end + 1
        else:
            frag_start_pos = 0
    else:
        # fallback: start of fragment is after <html> or 0
        idx_html = lower.find("<html")
        if idx_html != -1:
            idx_html_end = lower.find(">", idx_html)
            frag_start_pos = idx_html_end + 1 if idx_html_end != -1 else 0
        else:
            frag_start_pos = 0

    frag_end_pos = len(fragment)

    # Offsets measured in bytes of the entire payload after header
    preamble = "Version:0.9\r\n"
    # placeholder offsets will be replaced
    header = (preamble +
              "StartHTML:__________\r\n"
              "EndHTML:___________\r\n"
              "StartFragment:__________\r\n"
              "EndFragment:___________\r\n")
    header_bytes = header.encode("utf-8")
    # Compute real offsets
    start_html_off = len(header_bytes)
    html_bytes = fragment.encode("utf-8")
    end_html_off = start_html_off + len(html_bytes)
    # Compute fragment offsets relative to start_html_off
    # We need byte offsets for fragment start and end within the HTML payload
    frag_start_byte = start_html_off + len(fragment[:frag_start_pos].encode("utf-8"))
    frag_end_byte = start_html_off + len(fragment[:frag_end_pos].encode("utf-8"))

    # Fill header with zero-padded numbers (10 digits)
    header_filled = (preamble +
                     f"StartHTML:{start_html_off:010d}\r\n" +
                     f"EndHTML:{end_html_off:010d}\r\n" +
                     f"StartFragment:{frag_start_byte:010d}\r\n" +
                     f"EndFragment:{frag_end_byte:010d}\r\n")
    final = header_filled.encode("utf-8") + html_bytes
    return final

def copy_html_windows_pywin(html: str) -> bool:
    """Usa pywin32 para copiar HTML al clipboard como 'HTML Format'."""
    try:
        import win32clipboard
        import win32con
    except Exception as e:
        print(f"[debug] pywin32 not available: {e}", file=sys.stderr)
        return False

    cf_html = build_cf_html_fragment(html)  # bytes
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            # Register custom format
            cf = win32clipboard.RegisterClipboardFormat("HTML Format")
            win32clipboard.SetClipboardData(cf, cf_html)
        finally:
            win32clipboard.CloseClipboard()
        return True
    except Exception as e:
        print(f"[debug] win32clipboard failed: {e}", file=sys.stderr)
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass
        return False

# ---------- Orchestrator ----------
def copy_html_to_clipboard(html: str) -> bool:
    system = platform.system().lower()
    # macOS
    if system == "darwin":
        # 1) pbcopy-html.swift if present
        if copy_html_mac_pbcopy_swift(html):
            return True
        # 2) try pyobjc
        if copy_html_mac_pyobjc(html):
            return True
        # 3) fallback: plain pbcopy (won't set html type) - last resort
        if shutil.which("pbcopy"):
            try:
                subprocess.run(["pbcopy"], input=html.encode("utf8"), check=True)
                print("[warning] HTML pasted as plain text (pbcopy). LinkedIn editor may not preserve formatting.", file=sys.stderr)
                return True
            except Exception:
                pass
        return False

    # Linux
    if system == "linux":
        if copy_html_linux(html):
            return True
        # fallback to plain xclip/pbcopy-like (won't set mime)
        if shutil.which("xclip"):
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=html.encode("utf8"), check=True)
                print("[warning] Clipboard set but MIME type may not be text/html.", file=sys.stderr)
                return True
            except Exception:
                pass
        return False

    # Windows
    if system == "windows":
        if copy_html_windows_pywin(html):
            return True
        # fallback: set unicode text only
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, html)
            win32clipboard.CloseClipboard()
            print("[warning] Only plain text was set on Windows clipboard (no HTML MIME).", file=sys.stderr)
            return True
        except Exception:
            return False

    # Other OS: fallback plain copy if possible
    if shutil.which("pbcopy"):
        try:
            subprocess.run(["pbcopy"], input=html.encode("utf8"), check=True)
            print("[warning] Used pbcopy fallback; HTML type may not be set.", file=sys.stderr)
            return True
        except Exception:
            pass
    return False

# ---------- CLI ----------
def read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    else:
        with open(path, "r", encoding="utf8") as f:
            return f.read()

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown -> HTML and copy HTML to clipboard (text/html).")
    parser.add_argument("infile", help="Input markdown file path or '-' for stdin")
    parser.add_argument("--no-pandoc", action="store_true", help="Don't try pandoc even if present (force python-markdown).")
    parser.add_argument("--save-html", help="Save generated HTML to this file (for debugging).")
    parser.add_argument("--pbcopy-swift", help="Path to pbcopy-html.swift executable (macOS).")
    args = parser.parse_args()

    md = read_input(args.infile)
    # convert
    html = None
    if shutil.which("pandoc") and not args.no_pandoc:
        try:
            html = md_to_html_with_pandoc(md)
        except Exception as e:
            print(f"[info] pandoc conversion failed: {e}", file=sys.stderr)
    if html is None:
        try:
            html = md_to_html_with_python_md(md)
        except Exception as e:
            print(f"[error] markdown -> html conversion failed: {e}", file=sys.stderr)
            sys.exit(2)

    # Optionally save
    if args.save_html:
        with open(args.save_html, "w", encoding="utf8") as fh:
            fh.write(html)
        print(f"Saved HTML to {args.save_html}")

    # Try copy to clipboard
    ok = False
    system = platform.system().lower()
    if system == "darwin" and args.pbcopy_swift:
        ok = copy_html_mac_pbcopy_swift(html, swift_exec=args.pbcopy_swift)
    else:
        ok = copy_html_to_clipboard(html)

    if ok:
        print("HTML copied to clipboard (attempt succeeded). Paste into LinkedIn editor and verify formatting.")
        sys.exit(0)
    else:
        print("Failed to copy HTML to clipboard with MIME type. As fallback, saving HTML to a temporary file and opening it in default browser.", file=sys.stderr)
        # save to temp file and open
        fd, tmp = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, "w", encoding="utf8") as fh:
            fh.write(html)
        print(f"Saved HTML to {tmp}", file=sys.stderr)
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", tmp])
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", tmp])
            elif sys.platform.startswith("win"):
                os.startfile(tmp)
        except Exception as e:
            print(f"[debug] Could not open temp HTML file automatically: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
