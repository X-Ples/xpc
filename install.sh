#!/usr/bin/env bash

set -u

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

CYAN='\033[1;36m'; GREEN='\033[1;32m'; YEL='\033[1;33m'; RED='\033[1;31m'; NC='\033[0m'
info(){ printf "${CYAN}>${NC} %s\n" "$1"; }
ok(){   printf "${GREEN}✓${NC} %s\n" "$1"; }
warn(){ printf "${YEL}!${NC} %s\n" "$1"; }
die(){  printf "${RED}✗ %s${NC}\n" "$1" >&2; exit 1; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$DIR/xplus"

echo " X+ installer — installer v1.0.0"

if ! command -v python3 >/dev/null 2>&1; then
  die "python3 not found. Please install Python 3.8 or newer first."
fi
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "python3 $PYVER"

detect_layout() {
  if [ -f "$PKG/__init__.py" ]; then
    echo "A"
  elif [ -f "$DIR/__init__.py" ] && [ -f "$DIR/lexer.py" ] && [ -f "$DIR/compiler.py" ]; then
    echo "B"
  else
    echo "X"
  fi
}

LAYOUT="$(detect_layout)"

if [ "$LAYOUT" = "X" ]; then
  die "Cannot find the X+ sources here. Run this script from the folder that contains either 'xplus/__init__.py' (standard layout) or the package files themselves (e.g. ~/X+ with lexer.py, compiler.py, __init__.py inside)."
fi

if [ "$LAYOUT" = "B" ]; then
  info "Flat layout detected (package files sit next to install.sh)."
  info "Restructuring: moving Python modules into ./xplus/ ..."
  mkdir -p "$PKG" || die "Cannot create directory: $PKG (permission denied?)"
  moved=0
  for f in "$DIR"/*.py; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    if [ "$base" = "xpc.py" ]; then
      continue
    fi
    if [ -f "$PKG/$base" ]; then
      warn "  skip $base (already exists inside xplus/)"
      continue
    fi
    if mv -n "$f" "$PKG/" 2>/dev/null; then
      moved=$((moved + 1))
    else
      warn "  could not move $base (leaving it in place)"
    fi
  done
  ok "Restructured: $moved module(s) moved into ./xplus/"
  ok "Your folder now uses the standard X+ project layout (nothing was deleted)."
fi

[ "$(detect_layout)" = "A" ] || die "Restructuring failed — $PKG/__init__.py is still missing."
ok "Package found: $PKG"

write_launcher() {
  local target="$1" root="$2"
  cat > "$target" <<PYEOF
#!/usr/bin/env python3
# xpc — X+ toolchain launcher
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BAKED_ROOT = r"""$root"""

for _p in (_HERE, _BAKED_ROOT):
    if _p and os.path.isfile(os.path.join(_p, "xplus", "__init__.py")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
else:
    print("xpc: xplus/__init__.py not found (looked in %s and %s)" % (_HERE, _BAKED_ROOT), file=sys.stderr)
    print("     re-run install.sh inside the X+ project folder", file=sys.stderr)
    sys.exit(70)


def _main():
    import importlib
    import traceback

    candidates = (
        ("xplus.cli", "main"),
        ("xplus.cli", "cli"),
        ("xplus.cli", "main_cli"),
    )
    missing = []
    for modname, attr in candidates:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("xplus"):
                missing.append(modname)
                continue
            print("xpc: error while importing %s:" % modname, file=sys.stderr)
            traceback.print_exc()
            return 70
        except Exception:
            print("xpc: error while importing %s:" % modname, file=sys.stderr)
            traceback.print_exc()
            return 70
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn()
    print("xpc: could not locate a CLI entry point (tried: %s)" % ", ".join(missing), file=sys.stderr)
    print("     re-run install.sh inside the X+ project folder", file=sys.stderr)
    return 70


if __name__ == "__main__":
    sys.exit(_main())
PYEOF
  chmod +x "$target" 2>/dev/null || true
}

info "Writing launcher: $DIR/xpc"
write_launcher "$DIR/xpc" "$DIR"
ok "Created local launcher: ./xpc"

if [ -n "${TERMUX_VERSION:-}" ] && [ -n "${PREFIX:-}" ] && [ -d "$PREFIX/bin" ]; then
  GLOBAL_BIN="$PREFIX/bin"
  info "Termux detected — global command will go to \$PREFIX/bin"
else
  GLOBAL_BIN="$HOME/.local/bin"
fi

if { [ -d "$GLOBAL_BIN" ] || mkdir -p "$GLOBAL_BIN" 2>/dev/null; } && [ -w "$GLOBAL_BIN" ]; then
  write_launcher "$GLOBAL_BIN/xpc" "$DIR"
  ok "Installed global command: $GLOBAL_BIN/xpc"
  case ":$PATH:" in
    *":$GLOBAL_BIN:"*) : ;;
    *)
      warn "$GLOBAL_BIN is not on your PATH."
      warn "Enable it:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc  &&  source ~/.bashrc"
      ;;
  esac
else
  warn "Could not create $GLOBAL_BIN — use ./xpc from the project folder instead."
fi

if mkdir -p "$HOME/.xpc/libs" 2>/dev/null; then
  ok "Package cache ready: $HOME/.xpc/libs"
fi

info "Running smoke test ..."
if PYTHONPATH="$DIR" python3 -c "import xplus.compiler, xplus.cli; print('    import xplus.compiler, xplus.cli ... OK')"; then
  ok "Core modules import cleanly."
else
  die "Import test failed. Make sure ALL package files (__init__.py, lexer.py, parser.py, astnodes.py, tokens.py, typesys.py, xir.py, vm.py, compiler.py, cli.py, repl.py, resolver.py, errors.py, builtins.py) are inside ./xplus/ — then re-run install.sh."
fi

echo
info "Component check:"
if [ -d "$DIR/examples" ]; then
  ok "  examples/  found"
else
  warn "  examples/  missing — demo programs unavailable (drop an examples/ folder here to enable)"
fi
if [ -n "$(ls "$DIR"/libs/*.xp 2>/dev/null | head -n 1)" ] || [ -n "$(ls "$HOME"/.xpc/libs/*.xp 2>/dev/null | head -n 1)" ]; then
  ok "  stdlib     found"
else
  warn "  stdlib     missing — programs that import libraries need libs/*.xp (place them in ./libs/ or ~/.xpc/libs/)"
fi
if [ -d "$DIR/tests" ]; then
  ok "  tests/     found"
else
  warn "  tests/     missing — self-test suite unavailable"
fi

echo
printf "${GREEN}✓ X+ is ready.${NC}\n"
echo "  ./xpc repl                    interactive REPL"
echo "  ./xpc run examples/hello.xp   run a program"
echo "  xpc doctor                    environment check"