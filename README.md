<div align="center">

# X+ (X Plus)

**A safe, tensor-first programming language: compile-time matrix shape checking,
absolute null-safety, arena memory, autodiff, and a hardware world
(MMIO/asm/interrupts) inside the VM — shipped as a ready-to-run binary,
available on Linux, macOS, Windows, and Android/Termux.**

`xpc toolchain 1.1.0` · `language v0.1 (locked mode)` · `bytecode: .xbc (XBC1)` ·
`ecosystem: X-Ples/Liblary`

</div>

> **New in 1.1.0:** a faster Zig-based compiler (`xzc`), a Rust-core VM (`xpv`),
> tiny `.xbc` artifacts, and a tiny memory footprint. See
> [What's New in 1.1.0](#29-whats-new-in-110),
> [Performance & Benchmarks](#30-performance--benchmarks-110) and
> [Important Tips](#31-important-tips).
> Download: <https://github.com/X-Ples/xpc/releases/tag/binary>

---

## Table of Contents

1. [What is X+?](#1-what-is-x)
2. [Why X+](#2-why-x)
3. [Installation](#3-installation)
4. [Your First Program](#4-your-first-program)
5. [Syntax Tour (Locked Mode)](#5-syntax-tour-locked-mode)
6. [Type System](#6-type-system)
7. [Tensors: vec, mat, `x` Matmul & `.x` Dot Product](#7-tensors-vec-mat-x-matmul--x-dot-product)
8. [Pipeline Operator `|>`](#8-pipeline-operator-)
9. [comptime — Compute at Compile Time](#9-comptime--compute-at-compile-time)
10. [rational — Exact Fractions](#10-rational--exact-fractions)
11. [Safety: guard, Result, T?, match, and ?](#11-safety-guard-result-t-match-and-)
12. [arena — Per-Scope Memory](#12-arena--per-scope-memory)
13. [@grad & @tensor_core](#13-grad--tensor_core)
14. [SMV — System Magic Variables](#14-smv--system-magic-variables)
15. [The Hardware World Inside the VM](#15-the-hardware-world-inside-the-vm)
16. [Error System & Visual Diagnostics](#16-error-system--visual-diagnostics)
17. [XIR & the .xbc Bytecode](#17-xir--the-xbc-bytecode)
18. [CLI Reference: xpc (40 Subcommands)](#18-cli-reference-xpc-40-subcommands)
19. [Standard Library](#19-standard-library)
20. [Library Resolution System (Offline-First)](#20-library-resolution-system-offline-first)
21. [Tutorial: Building a Private Library](#21-tutorial-building-a-private-library)
22. [Tutorial: Publishing to X-Ples/Liblary (Fork + PR)](#22-tutorial-publishing-to-x-plesliblary-fork--pr)
23. [Testing](#23-testing)
24. [Project Structure](#24-project-structure)
25. [Real-World Uses](#25-real-world-uses)
26. [Troubleshooting & FAQ](#26-troubleshooting--faq)
27. [Roadmap v0.5](#27-roadmap-v05)
28. [Ecosystem & Contributing](#28-ecosystem--contributing)
29. [What's New in 1.1.0](#29-whats-new-in-110)
30. [Performance & Benchmarks (1.1.0)](#30-performance--benchmarks-110)
31. [Important Tips](#31-important-tips)

---

## 1. What is X+?

X+ (pronounced *"X Plus"*) is a compiled programming language: `.xp` source
code is turned into **XIR** (*X+ Intermediate Representation*, SSA-flavored),
packaged as **`.xbc` bytecode**, and executed by the **XVM** (*X+ Virtual
Machine*). The toolchain is distributed as a **prebuilt, self-contained binary**
(`xpc`) — download it from the
[GitHub release page](https://github.com/X-Ples/xpc/releases/tag/binary), put it
on your `PATH`, and you are done. No language runtime or package manager is
required, so X+ runs on laptops, servers, Raspberry Pi boards, and even Android
phones through Termux.

X+ is built on three core philosophies:

**First, *safety from compile time*.** Errors that can be detected early must
not wait until runtime. The most iconic example: multiplying
`mat[2,3] x mat[4,2]` is **rejected by the compiler** with
`E204 Dimension Mismatch` — before the program ever runs. Integers are
*checked by default* (overflow triggers panic `E301` instead of silently
producing a strange number; need explicit wrapping? use `+% -% *%`), array
access is bounds-checked, and `null` cannot sneak in thanks to the `T?` type.

**Second, *math as a first-class citizen*.** `vec2`, `mat2`, and `mat[2,3]` are
built-in primitive types; the `x` operator multiplies matrices; `.x` computes
the dot product; `rational` does exact fraction arithmetic; `comptime` computes
lookup tables at compile time; and `@grad` computes numeric derivatives of
functions — all with no extra installation.

**Third, *a safe hardware world*.** X+ brings microcontroller-style programming
into the VM: `register GPIOA_ODR: u32 @ 0x40020014;` declares a type-safe MMIO
register (no pointer casts!), `asm! { "nop"; }` fires low-level instructions,
`@interrupt(vector::EXTI0)` registers an ISR, and
`gpio::write(2, gpio::HIGH)` lights up a virtual LED — all executed on top of a
fully-checked XVM.

X+ supports **two syntax modes**: **legacy** mode (macro blocks
`@show(sys::out) { ... }` with slot injection, `yield` as return), which remains
fully supported for compatibility, and **locked** mode
(`show(sys::out, "...")` + `return`), which is the official mode. This entire
document uses locked mode.

> **Note:** the formal specification (EBNF, XIR instruction set, XBC format,
> SMV table) lives in **`SPECS.md`**. The **`examples/`** folder contains 20
> ready-to-run sample programs — every code snippet in this document is taken
> from real code that compiles.

---

## 2. Why X+

| # | Strength | Description |
|---|----------|-------------|
| 1 | **Compile-time shape checking** | `mat[2,3] x mat[4,2]` is rejected with `E204` — linear-algebra mistakes are caught before runtime |
| 2 | **Checked integers by default** | Overflow → panic `E301`; explicit wrapping operators `+% -% *%` are available |
| 3 | **Bounds checking** | Out-of-range array/tensor access → `E303`, not memory corruption |
| 4 | **Absolute null-safety** | `T?` + `match { Some(x) =>, None => }` — no null-class segfaults |
| 5 | **`Result<T,E>` + `?`** | Explicit, zero-overhead error propagation, Rust style |
| 6 | **`guard`** | Clean early preconditions; even used by the stdlib for max/min loops |
| 7 | **`comptime` (CTFE)** | Lookup tables computed by the compiler — 0 ms at runtime; 1,000,000-step limit (`E220`) |
| 8 | **`rational`** | `1/3 + 2/7 = 13/21` exactly — no floating-point error |
| 9 | **`@grad`** | One-line numeric autodiff: `@grad(loss_func)` produces the derivative function |
| 10 | **`@tensor_core`** | Kernel blocks are offloaded to the accelerator path (SIMD, simulated) |
| 11 | **±40 active SMVs** | Hardware/runtime telemetry right in your code: `@simd_width`, `@core_count`, `@arena_peak`, ... |
| 12 | **`arena`** | All allocations in a block are released at once on scope exit — 0 fragmentation |
| 13 | **Pipeline `\|>`** | `raw_signal \|> normalize \|> relu \|> scale(2.0)` — data flows left to right |
| 14 | **MMIO / `asm!` / `@interrupt` / GPIO** | A virtual microcontroller lab that is type-safe, with no pointer casts |
| 15 | **Coded visual errors** | E1xx/E2xx/E3xx + W1xx warnings, with a caret pointing at the position |
| 16 | **Offline-first** | 4-level resolver + `xpc.lock` lockfile (SHA-256); internet is optional |
| 17 | **Single prebuilt binary** | Download and run — nothing to build or install; Linux/macOS/Windows/Termux |
| 18 | **40-subcommand CLI** | run/check/lint/ast/ir/disasm/build/bench/fmt/watch/install/audit/simulate/dll |
| 19 | **Open ecosystem** | Central public libraries at `X-Ples/Liblary`; just fork + PR |
| 20 | **Fast compile & tiny output (1.1.0)** | ~41x faster than `g++` on a 1000-line module; an 83-byte `.xbc` for hello world — see [Section 30](#30-performance--benchmarks-110) |

In short: X+ combines Rust-style *type safety*, Python-style scripting
convenience, and an embedded-style hardware lab — in one small, consistent
language.

---

## 3. Installation

X+ is distributed as **prebuilt binaries**. All releases are published at:

**<https://github.com/X-Ples/xpc/releases/tag/binary>**

### 3.1 Download the binary

1. Open the [release page](https://github.com/X-Ples/xpc/releases/tag/binary).
2. Under **Assets**, download the binary that matches your operating system and
   CPU architecture.
3. Rename it to `xpc` (or `xpc.exe` on Windows) if needed.

### 3.2 Put it on your `PATH`

**Linux / macOS:**

```bash
chmod +x xpc
mkdir -p ~/.local/bin
mv xpc ~/.local/bin/xpc          # make sure ~/.local/bin is on your PATH
```

**Android / Termux:**

```bash
chmod +x xpc
mv xpc $PREFIX/bin/xpc           # $PREFIX/bin is already on the Termux PATH
```

**Windows:** move `xpc.exe` to a folder of your choice and add that folder to
your `PATH` environment variable.

### 3.3 Verify

```bash
xpc version                 # CLI + compiler + target versions (expect 1.1.0)
xpc doctor                  # toolchain health diagnosis
xpc cpu-info                # detect AVX-512 / NEON / FMA3
xpc run examples/hello.xp   # run your first example
```

> The `examples/` folder and the standard library (`libs/`) are part of the
> project repository. If you only downloaded the binary, the stdlib is bundled
> inside it (resolution level 3, see [Section 20](#20-library-resolution-system-offline-first)),
> and you can write your own `.xp` files anywhere.

### 3.4 Requirements

- A supported operating system (Linux, macOS, Windows, or Android/Termux) on a
  supported CPU architecture — see the assets listed on the release page.
- **Nothing else.** No language runtime, no third-party dependencies.

### 3.5 Updating

```bash
xpc self-update             # update xpc itself
```

You can always re-download the latest binary (1.1.0) from the
[release page](https://github.com/X-Ples/xpc/releases/tag/binary) as well.

---

## 4. Your First Program

Create a file named `hello.xp`:

```rust
main => int32 {
    name := "World";
    speed := 999;

    show(sys::out, "Hello, {name}! Speed: {speed}\n");
    show(sys::out, "Precision format: {speed as f64:.2}\n");

    yield 0;
}
```

Run it:

```bash
xpc run hello.xp
```

Output:

```
Hello, World! Speed: 999
Precision format: 999.00
```

Notice three signature X+ traits:

1. **`main => int32 { ... }`** — an implicit entry point (you may add the
   `@entry` annotation; a `void` return type is also valid).
2. **`show(sys::out, "...")`** — output goes to the `sys::out` channel, with
   `{expression}` interpolation resolved **at compile time** (zero runtime
   format errors; variables are validated by the type checker).
3. **`{speed:.2}` and `{speed as f64}`** — precision formatting and explicit
   casts directly inside the format string.

Three ways to execute X+ code:

| Way | Command | When to use |
|-----|---------|-------------|
| Run directly | `xpc run hello.xp` | Everyday use; compile to RAM (tmpfs) + execute |
| AOT compile | `xpc build hello.xp` → `xpc hello.xbc` | Distributing bytecode; `xpc app.xbc` runs directly |
| REPL | `xpc repl` | Quick experiments, JIT in RAM |

---

## 5. Syntax Tour (Locked Mode)

### 5.1 Variables: `:=` and type annotations

```rust
name := "Dimas";            // type inference
speed: f64 = 999.0;         // with an explicit annotation
data := [1.0, 2.0, 3.0];    // array literal
zeros := [0.0; 4];          // fill: [0.0, 0.0, 0.0, 0.0]
```

Two declaration forms: `ident := expr;` (inferred) and `ident: type = expr;`
(explicit). **Statements end with `;`.** Global constants use `const`
(immutable — violations are rejected by the compiler with `E208`):

```rust
const GRAVITY: f64 = 9.80665;
const MAX_BUFFER: int32 = 4096;
```

### 5.2 Functions

```rust
// function with full types
fn safe_divide(a: f64, b: f64) -> f64 {
    guard b != 0.0 else {
        return 0.0;
    }
    return a / b;
}

// implicit function: untyped params, return via yield (legacy alias)
fn normalize(data) {
    total := sum(data);
    result := [];
    for x in data {
        push(result, x / total);
    }
    yield result;
}
```

`return` (locked) and `yield` (legacy alias) are both valid; locked mode
recommends `return` — `yield` is reserved to return as a generator in v0.5.

### 5.3 Branching, loops, match

```rust
if blink > 5 { break; }             // plain block
if x > 0.0 { push(result, x); } else { push(result, 0.0); }

for i in 0..4 { ... }               // exclusive range 0,1,2,3
for i in 1..=6 { ... }              // inclusive range
loop {                              // conditionless loop + break
    blink = blink + 1;
    if blink > 5 { break; }
}

match result {
    Ok(v)  => { show(sys::out, "OK: {v}\n"); }
    Err(e) => { show(sys::out, "failed: {e}\n"); }
}

match u1 {
    Some(name) => { show(sys::out, "Found: {name}\n"); }
    None       => { show(sys::out, "Not found\n"); }
}
```

### 5.4 Structs

```rust
struct Player {
    name: string;
    speed: f64;
    level: int32;
}

main => int32 {
    p1 := Player {
        name: "Dimas",
        speed: 999.0,
        level: 99
    };
    p1.speed = p1.speed + 1.0;      // field mutation
    show(sys::out, "{p1.name} speed {p1.speed}\n");
    yield 0;
}
```

### 5.5 Legacy mode (compatibility)

Older programs use the `@show` macro block with slot injection — even with
per-slot type annotations:

```rust
@show(sys::out) {
    "Player {0} has speed {1}\n";
    0::string => p1.name;
    1::f64    => p1.speed;
};
```

The parser accepts both styles, but **locked is the official mode** for new code.

---

## 6. Type System

| Family | Types | Notes |
|--------|-------|-------|
| Signed integers | `i8` `i16` `i32` `i64` | checked by default (overflow → `E301`) |
| Unsigned integers | `u8` `u16` `u32` `u64` `usize` | ideal for registers & MMIO addresses |
| Common aliases | `int32` `int64` | shortcuts frequently used in examples |
| Floats | `f32` `f64` | f32 for graphics/games, f64 for science |
| Logic/text | `bool` `string` `void` | |
| Exact | `rational` | exact fractions; `decimal256` for finance |
| Tensors | `vec2` `vec3` `vec4`, `mat2` `mat3` `mat4`, `mat[2,3]`, `tensor[dims]` | built-in primitives + shape checking |
| Fixed array | `[f64; 4]` | size known at compile time |
| Null-safe | `T?` | `Some(v)` or `None` |
| Error-safe | `Result<T,E>` | `Ok(v)` or `Err(e)` |
| User-defined | `struct`, `enum`, generic `Result<T,E>` | `struct Player { ... }` |

**Key principle:** `None` and errors **cannot** sneak into ordinary types.
A function that can fail returns `Result<T,E>`; one that can be empty returns
`T?`. The compiler forces you to handle both — which is why X+ programs are
nearly free of "unexpected" crashes.

---

## 7. Tensors: vec, mat, `x` Matmul & `.x` Dot Product

### 7.1 Creating tensors — array literals in a typed context

```rust
W: mat2 = [[0.2, 0.4], [0.5, 0.8]];     // 2x2 matrix
X: vec2 = [1.0, 2.0];                   // 2-dimensional vector

A23: mat[2, 3] = [[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]];     // generic 2x3 matrix
B32: mat[3, 2] = [[7.0,  8.0],
                  [9.0,  10.0],
                  [11.0, 12.0]];        // 3x2 matrix
```

Matrices and vectors are **built-in primitive types** — not a library, not a
class. The literal is an array-of-arrays; the type checker validates its shape.

### 7.2 Matrix multiplication with `x` — checked at compile time

```rust
Y := W x X;                 // matmul mat2 x vec2 (in XIR: tensor.mul.mat2)
C := A x B;                 // matmul mat2 x mat2
C22 := A23 x B32;           // [2x3] x [3x2] -> [2x2] : COMPILES
```

Note: both sides of `x` must be **tensors** (mat×mat, mat×vec, vec×mat).
Scalar multiplication uses `*` (broadcast — see 7.4).

If the shapes do not match, **compilation fails immediately**:

```rust
A: mat[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
B: mat[4, 2] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
C := A x B;   // <-- E204: [2x3] x [4x2]: columns of A (3) != rows of B (4)
```

The error message is visual and explains the fix:

```
[X+ Type Error]: E204 Dimension Mismatch in Matrix Multiplication: [2 x 3] x [4 x 2]
 --> tensor_shape_err.xp:13:10
  |
13|     C := A x B;
  |          ^--------------------------------------
  = columns of A (3) must equal rows of B (4)
```

This is X+'s most iconic feature: **linear-algebra mistakes are caught at
compile time, 0 ms of runtime.**

### 7.3 Dot product `.x` — and the spacing rule

| Writing | Meaning |
|---------|---------|
| `v1 .x v2` (separated by spaces) | **Dot product** (in XIR: `bin dot`) |
| `v.x` (attached) | **Field access** to a field named `x` |

```rust
v1: vec4 = [1.0, 2.0, 3.0, 4.0];
v2: vec4 = [2.0, 2.0, 2.0, 2.0];

d := v1 .x v2;        // dot product -> 20.0
```

This rule is consistent and documented in SPECS.md: **space = dot-product
operator; attached = member access.**

### 7.4 Scalar broadcast & elementwise

Elementwise vec operations (`+ - * /`) accept a scalar and *broadcast* it
across all elements — **use `*` for scalar multiplication** (the `x` operator is
strictly for tensor × tensor matmul):

```rust
g := v1 * 2.0;        // scalar broadcast across all elements -> [2.0, 4.0, 6.0, 8.0]
h := v1 + v2;         // elementwise
d := a - b;           // elementwise (used by the stdlib & distance examples)
```

### 7.5 Important note on the `x` token

The `x` token is the matmul operator **only in infix position** (`A x B`).
Everywhere else it is an ordinary identifier — both the old-style `let x = 5`
and `x := 5` remain valid.

---

## 8. Pipeline Operator `|>`

Data flows left to right; the piped value becomes the **first argument** of the
next function:

```rust
raw_signal := [0.1, -0.5, 3.2, -1.2];

raw_signal |> normalize |> relu |> scale(2.0) |> show_vec;

// Classic math pipelines:
s := 30.0 |> math::deg_to_rad |> math::sin;
mag := (3.0 ^ 2.0 + 4.0 ^ 2.0) |> math::sqrt;      // |(3,4)| = 5.0
```

Pipelines make data transformations read like a flow diagram — a great fit for
signal processing and mini ML pipelines. Full example: `examples/pipeline.xp`.

---

## 9. comptime — Compute at Compile Time

`const` + a `comptime { ... }` block is fully evaluated by the compiler — the
result is embedded in the bytecode, **0 ms at runtime**:

```rust
const SIN_TABLE: [f64; 4] = comptime {
    arr := [0.0; 4];
    for i in 0..4 {
        arr[i] = __builtin_fast_sin((i as f64) * 90.0 * @pi / 180.0);
    }
    yield arr;
};

const FACTS: [f64; 6] = comptime {
    arr := [1.0; 6];
    acc := 1.0;
    for i in 1..6 {
        acc = acc * (i as f64);
        arr[i] = acc;
    }
    yield arr;
};
```

CTFE rules (per the LOCKED decisions in SPECS.md):

- **Pure functions only** — no I/O, network, randomness, or side effects.
- **Step limit of 1,000,000 steps** — violations stop compilation with `E220`
  (instead of hanging forever).
- CTFE float determinism uses the host's reference libm (v0.1); v0.5 uses an
  internal software-based libm so builds are reproducible.

When comptime is useful: trigonometric tables, factorial lookups, buffer/kernel
size expansion, and early validation of numeric invariants.

---

## 10. rational — Exact Fractions

The `rational` type computes fractions **without rounding** — suitable for
cryptography, finance (`decimal256`), and number-theory research:

```rust
main => int32 {
    a: rational = 1/3;
    b: rational = 2/7;

    exact_sum := a + b;      // stays an exact fraction: 13/21
    show(sys::out, "1/3 + 2/7 = {exact_sum}\n");

    fa := 1.0 / 3.0;
    fb := 2.0 / 7.0;
    show(sys::out, "f64     : {fa + fb:.17}  (has float error)\n");
    show(sys::out, "rational: exactly 13/21 ✓\n");
    yield 0;
}
```

Just annotate with `rational` — the literal `1/3` in that typed context
automatically becomes an exact fraction. The operations `+ - * /` between
rationals yield a rational again.

---

## 11. Safety: guard, Result, T?, match, and ?

### 11.1 `guard` — early preconditions

`guard condition else { ... }` asserts: *"this condition must hold; otherwise,
leave here."* Even the stdlib uses it to find max/min:

```rust
fn max(data) -> f64 {
    largest := data[0];
    for x in data {
        guard x <= largest else { largest = x; };
    }
    yield largest;
}
```

### 11.2 `Result<T,E>` — success or failure, zero-overhead

```rust
fn safe_divide(a: f64, b: f64) -> Result<f64, string> {
    guard b != 0.0 else {
        return Err("division by zero");
    }
    return Ok(a / b);
}
```

### 11.3 The `?` operator — automatic propagation

```rust
fn compute_ratio(a: f64, b: f64) -> Result<f64, string> {
    value := safe_divide(a, b)?;      // Err -> immediately propagated to the caller
    return Ok(value * 100.0);
}
```

### 11.4 `match` — explicit handling

```rust
r2 := safe_divide(1.0, 0.0);
match r2 {
    Ok(v)  => { show(sys::out, "impossible\n"); }
    Err(e) => { show(sys::out, "1 / 0 -> Err({e})\n"); }
}
```

### 11.5 `T?` — optional, free of nullptr/segfault

```rust
fn find_user(id: int32) -> string? {
    if id == 1 { return Some("Dimas"); }
    if id == 2 { return Some("Rina"); }
    return None;
}

u9 := find_user(9);
match u9 {
    Some(name) => { show(sys::out, "Found: {name}\n"); }
    None       => { show(sys::out, "ID 9: not found (no segfault!)\n"); }
}
```

Runtime panics that still guard you even when everything above passes the type
check: `E301` overflow, `E302` div-zero, `E303` bounds, `E304` match
fall-through, `E305` unwrap-Err, `E306` cast overflow.

---

## 12. arena — Per-Scope Memory

`arena { ... }` is a memory scope: every allocation inside it is tracked by the
compiler and **released all at once when the block closes** (RAII-style, 1
cycle, 0 fragmentation):

```rust
main => int32 {
    arena {
        batch := make_batch(10);
        var_ := statistics::variance(batch);
        avg := statistics::mean(batch);
        show(sys::out, "mean = {avg}\n");
    }   // <- the entire batch allocation vanishes cleanly right here

    show(sys::out, "arena total : {@arena_total} bytes\n");
    show(sys::out, "arena peak  : {@arena_peak} bytes\n");
    show(sys::out, "stack depth : {@call_depth} frame(s)\n");
    yield 0;
}
```

In XIR, an arena is expressed as an `arena.enter` / `arena.exit` pair with a
cleanup guarantee. The *arena-first ownership* rule: arrays `[]` and strings are
allocated in the local arena scope by default; **globals may only be
`const`/comptime** (a mutable global is compile error `E208`). Memory telemetry
is available via SMVs: `@arena_free`, `@arena_total`, `@arena_peak`,
`@stack_usage`, `@heap_allocated`.

---

## 13. @grad & @tensor_core

### 13.1 `@grad(loss_func)` — one-line numeric autodiff

`@grad` takes a function name and produces its **derivative function**
(numerically):

```rust
fn loss_func(x: f64) -> f64 {
    return x ^ 2.0 + 3.0;
}

main => int32 {
    grad_loss := @grad(loss_func);
    d_val := grad_loss(4.0);
    show(sys::out, "@grad(x^2+3) at x=4.0 = {d_val:.4} (exact: 8.0)\n");
    yield 0;
}
```

At the XIR level, @grad functions are invoked through `callv` (dynamic call).
Useful for learning calculus, testing loss functions, and mini machine-learning
pipelines without any external framework.

### 13.2 `@tensor_core { ... }` — offload to an accelerator

The tensor annotation is used as a **block** — the block's contents execute
through the accelerator path (SIMD, simulated):

```rust
@tensor_core {
    W: mat2 = [[0.2, 0.4], [0.5, 0.8]];
    X: vec2 = [1.0, 2.0];
    Y := W x X;
    show(sys::out, "tensor_core output = {Y}\n");
}
```

Other tensor block families in the spec: `@gpu`, `@deterministic`, `@resilient`,
`@enclave`. Combine them with the SMVs `@simd_width` / `@tensor_backend` to tune
kernel size dynamically (see `examples/vec_simd.xp`).

---

## 14. SMV — System Magic Variables

**SMVs (*System Magic Variables*)** are special `@`-prefixed variables supplied
by the environment — never declared, read-only, and usable in any expression
(including inside `show` format strings and `comptime` contexts).

### 14.1 Implemented SMV table (v0.1)

| Group | SMV |
|-------|-----|
| **Math** | `@pi` `@euler` `@tau` `@phi` |
| **Hardware/SIMD** | `@simd_width` `@simd_lane_count` `@is_simd_active` `@core_count` `@cpu_freq_mhz` `@thread_id` |
| **Telemetry** | `@time_ns` `@uptime_ms` `@random_entropy` |
| **Arena/Memory** | `@arena_free` `@arena_total` `@arena_peak` `@stack_usage` `@heap_allocated` `@is_unsafe` |
| **Build** | `@compiler_ver` `@target_arch` `@target_os` `@build_profile` `@opt_level` `@tensor_backend` `@matrix_precision` `@ctfe_active` `@git_commit` `@build_timestamp` |
| **Source/Debug** | `@file` `@line` `@col` `@function` `@call_depth` |
| **App/AI** | `@app_name` `@environment` `@timezone` `@locale` `@log_level` `@terminal_width` `@debug_mode` `@batch_size` `@epoch_index` `@learning_rate` `@loss_value` |

SMVs outside the table (±120 more in the spec: network, HTTP, aerospace,
quantum groups, etc.) are rejected by the compiler with **`E120`** plus a
nearest-name suggestion. The v0.5 roadmap fills them in as backends become
available.

### 14.2 A real usage example

```rust
main => int32 {
    show(sys::out, "=== Runtime Telemetry & Hardware Context ===\n");
    show(sys::out, "  @simd_width    = {@simd_width} bit\n");
    show(sys::out, "  @simd_lane     = {@simd_lane_count} lane f64/clock\n");
    show(sys::out, "  @core_count    = {@core_count}\n");
    show(sys::out, "  @time_ns       = {@time_ns}\n");

    show(sys::out, "=== Compiler Environment ===\n");
    show(sys::out, "  @compiler_ver  = {@compiler_ver}\n");
    show(sys::out, "  @target_arch   = {@target_arch}\n");
    show(sys::out, "  @build_profile = {@build_profile}\n");
    show(sys::out, "  @tensor_backend= {@tensor_backend}\n");

    show(sys::out, "  @pi = {@pi:.6}, @euler = {@euler:.6}\n");
    show(sys::out, "  @phi (golden ratio) = {@phi:.6}\n");
    yield 0;
}
```

Try it yourself: `xpc run examples/smv_demo.xp`. For the CPU/VM info that makes
up the dynamic SMV values: `xpc cpu-info` (AVX-512 / NEON / FMA3 detection).

---

## 15. The Hardware World Inside the VM

All "hardware" is simulated — but the **semantics are serious**: real register
addresses, ISRs with vectors, and the most fundamental CPU instructions.

### 15.1 MMIO with `register` — type-safe, no pointer casts

```rust
register GPIOA_ODR: u32 @ 0x40020014;
register GPIOA_BSRR: u32 @ 0x40020018;

main => void {
    GPIOA_ODR = 0b00000001;             // write a register like an ordinary variable!
    show(sys::out, "GPIOA_ODR read back: {GPIOA_ODR}\n");

    GPIOA_ODR = GPIOA_ODR + 2;          // checked arithmetic on u32
    show(sys::out, "GPIOA_ODR + 2 = {GPIOA_ODR}\n");
    yield 0;
}
```

The declaration is `register NAME: type @ 0xADDRESS;` — the VM simulates the bus
and validates accesses; no raw pointers slip through.

### 15.2 `asm!` — inline assembly inside `unsafe`

```rust
unsafe {
    asm! {
        "nop";   // No Operation instruction
        "cli";   // Clear Interrupt Flag
    };
}
```

The body of `asm!` is a list of string statements. The `unsafe` block marks an
area outside the normal safety guarantees (see also the SMV `@is_unsafe`). Pair
it with `xpc disasm` to see how your X+ code is lowered into instructions.

### 15.3 `@interrupt(vector::NAME)` — ISR registration

```rust
@interrupt(vector::EXTI0)
fn button_pressed_handler() => void {
    gpio::toggle(13);
}
```

### 15.4 Board simulator

```bash
xpc run examples/iot_led.xp                          # normal execution
xpc simulate examples/iot_led.xp --board=esp32       # on-RAM embedded emulator
```

The `iot_led.xp` example blinks GPIO 2 (the ESP32's built-in LED) via
`gpio::mode(2, gpio::OUTPUT)` → `gpio::write(2, gpio::HIGH/LOW)` →
`time::delay_ms(100)` — a microcontroller lab without a single wire.

---

## 16. Error System & Visual Diagnostics

All X+ errors are coded and categorized:

| Code | Category | Example |
|------|----------|---------|
| **E1xx** | Syntax | unexpected token, unbalanced parentheses |
| **E120** | Unknown SMV | `@does_not_exist` → nearest-name suggestion |
| **E2xx** | Type/compile | `E204` matmul dimensions, `E208` mutable global |
| **E220** | CTFE | 1,000,000-step limit exceeded |
| **E3xx** | Runtime | `E301` overflow, `E302` div-zero, `E303` bounds, `E304` match fall-through, `E305` unwrap-Err, `E306` cast overflow |
| **W1xx** | Warning | from `xpc lint` |

The diagnostic format is visual — source line, a caret pointing at the column,
the code, and a help message:

```
[X+ Type Error]: E204 Dimension Mismatch in Matrix Multiplication: [2 x 3] x [4 x 2]
 --> tensor_shape_err.xp:13:10
  |
13|     C := A x B;
  |          ^--------------------------------------
  = columns of A (3) must equal rows of B (4)
```

Runtime protections that work automatically: checked integers (explicit wrapping
`+% -% *%` when truly intended), bounds checking, and the XVM stack guard. CLI
errors exit with code `70`, making them easy to catch in CI scripts.

---

## 17. XIR & the .xbc Bytecode

The X+ compilation pipeline:

```
hello.xp ─▶ lexer ─▶ parser (AST) ─▶ type checker ─▶ XIR (SSA) ─▶ .xbc ─▶ XVM
```

**XIR** (*X+ Intermediate Representation*) — SSA-flavored (temps `%N`, block
variables `%v_name`), first-class tensors, explicit arena markers,
backend-agnostic (ready to be lowered to Cranelift/LLVM in v0.5). Instruction
categories:

| Category | Instructions (examples) |
|----------|-------------------------|
| Dataflow | `const` `mov` `bin` `un` `cast` `smv` |
| Tensor | `bin matmul` `bin dot`, elementwise vec, scalar broadcast |
| Control | `label` `jmp` `jz` `jnz` `ret` `panic` |
| Calls | `call @fn(args)` `callv` (dynamic: fnref/@grad) `native` |
| Output | `show chan, fmt, args, specs` |
| Memory | `array` `fill_array` `index_get` `index_set` `struct` `member` `set_member` |
| Monadic | `wrap_ok` `wrap_err` `wrap_some` `wrap_none` `propagate` (=`?`) `extract` |
| Arena | `arena.enter` `arena.exit` (RAII) |
| Concurrency | `task.spawn` `task.wrap` `task.unwrap` (eager async) |
| Metal | `unsafe.enter/exit` `asm` `reg_read` `reg_write` `verify` |
| Trace | `line N` (source mapping for panic reports) |

Peek inside yourself:

```bash
xpc ast hello.xp       # dump the AST (JSON)
xpc ir hello.xp        # dump XIR
xpc asm hello.xp       # illustrative mapping to native assembly (SIMD)
xpc disasm hello.xbc   # disassemble bytecode
```

**`.xbc`** — a portable bytecode container:

```
offset  size  field
0       4     magic  "XBC1"
4       4     version u8, flags u8, reserved u16   (LE)
8       4     header_len u32 (LE) — JSON header
12      N     JSON header: xpc_version, target, fast_math, opt_level,
              build_timestamp, structs, enums, registers
12+N    4     body_len u32 (LE)
16+N    M     body: zlib(JSON { funcs, consts, structs, enums, registers })
```

A `.xbc` file can be executed directly: `xpc app.xbc` (a `.xbc` file argument is
automatically forwarded to `run --no-artifact` mode).

> **1.1.0:** in the `xzc` / `xpv` pair, a `.xbc` module stores only constants,
> native names, opcode streams and a CRC32 trailer. A hello-world module is
> just **83 bytes** — see [Section 30](#30-performance--benchmarks-110).

---

## 18. CLI Reference: xpc (40 Subcommands)

`xpc` is the single entry point to the whole toolchain — **40 subcommands** in 7
groups. Global flags: `--version`, `-s/--silent`, `--no-color`, `--target`,
`--tmp-dir`, `--no-cache`.

### Group: Compile & Execute

| Command | Function |
|---------|----------|
| `xpc run <file.xp>` | Compile to RAM (tmpfs .xbc) + execute immediately |
| `xpc check <file.xp>` | Validate syntax + types (output → /dev/null, instant) |
| `xpc lint <file.xp>` | Check + additional static analysis |
| `xpc ast <file.xp>` | Dump the Abstract Syntax Tree (JSON) |
| `xpc ir <file.xp>` | Dump XIR (SSA-flavored, backend-agnostic) |
| `xpc asm <file.xp>` | Illustrative mapping to native assembly (SIMD) |
| `xpc disasm <file.xbc>` | Disassemble a .xbc file |
| `xpc compile <file.xp>` | AOT compile → .xbc in tmpfs (not the project folder) |
| `xpc build <file.xp>` | Build .xbc to an explicit path (distribution) |

### Group: Measurement

| Command | Function |
|---------|----------|
| `xpc time <file.xp>` | Measure compile vs. execution time |
| `xpc bench <file.xp>` | ns-level micro-benchmark (`--runs N`) |

### Group: Development

| Command | Function |
|---------|----------|
| `xpc init` | Initialize an X+ project in this directory |
| `xpc new <name>` | mkdir + init a new project |
| `xpc fmt <file.xp>` | Reformat .xp code to the standard style |
| `xpc watch <file.xp>` | Hot-reload: recompile on every file change |
| `xpc repl` | Interactive X+ REPL (JIT in RAM) |

### Group: Package Manager

| Command | Function |
|---------|----------|
| `xpc install <lib>` | Download a library from X-Ples/Liblary |
| `xpc uninstall <lib>` | Remove a library from the local cache |
| `xpc update <lib>` | Update a library to the latest version |
| `xpc list` | List installed libraries |
| `xpc search <keyword>` | Search modules in the registry |
| `xpc info <lib>` | Package details in the registry |
| `xpc purge` | Delete the entire library cache |
| `xpc audit` | Audit outdated/corrupt local libraries |

### Group: Environment & Diagnostics

| Command | Function |
|---------|----------|
| `xpc env` | Active xpc configuration |
| `xpc doctor` | Toolchain health diagnosis |
| `xpc cpu-info` | Detect AVX-512 / NEON / FMA3 |
| `xpc clean` | Clean the tmpfs build |
| `xpc cache-size` | Size of the library cache + tmpfs |
| `xpc flags` | Active internal optimizer flags |
| `xpc tune` | Per-CPU-microarchitecture tuning (`--arch`) |
| `xpc version` | CLI + compiler + target versions |
| `xpc self-update` | Update xpc itself |

### Group: SIMD & Memory

| Command | Function |
|---------|----------|
| `xpc vectorize <file>` | Which loops can be SIMD-vectorized |
| `xpc sim-profile <file>` | Vectorization efficiency profile |
| `xpc cache-fit <file>` | Check fit in the L1 data cache |
| `xpc mem-check <file>` | Strict ownership + leak audit |
| `xpc bounds-check <file>` | Static out-of-bounds audit |

### Group: IoT

| Command | Function |
|---------|----------|
| `xpc simulate <file>` | On-RAM embedded simulator (`--board=esp32`) |

**A common daily workflow:**

```bash
xpc check main.xp      # 1. quick validation (no execution)
xpc lint main.xp       # 2. extra static analysis
xpc run main.xp        # 3. run
xpc build main.xp      # 4. produce a .xbc to distribute
```

---

## 19. Standard Library

The X+ standard library is **written 100% in X+ itself** (the `libs/` folder) —
you can read the source and use it as a style reference for writing libraries.
Official entries are registered in `libs/registry.toml`:

| Library | Version | Official description |
|---------|---------|----------------------|
| `math` | 2.8.1 | Fast-math trigonometry, logarithms & linear algebra (SIMD-accelerated: AVX-512 / NEON) |
| `strng` | 1.4.0 | String utilities: transform, query & zero-copy slice |
| `string` | 1.4.0 | Official alias for strng.xp (long-name compatibility) |
| `time` | 1.1.3 | Delay, timestamp & nanosecond-precision clock |
| `statistics` | 0.9.2 | Basic statistics: mean, variance, std — vectorized loops |
| `gpio` | 0.5.0 | Bare-metal GPIO for ESP32/STM32 + board simulator |
| `task` | 0.3.0 | Task-based concurrency & green threads (M:N scheduler in v0.5) |

Library consumption example (from `examples/library_user.xp`):

```rust
using = <<math.xp>>
using = <<strng.xp>>
using = <<statistics.xp>>

main => int32 {
    // math:: — SIMD-accelerated trigonometry
    s := math::sin_deg(30.0);
    show(sys::out, "math::sin_deg(30) = {s:.4}\n");

    // strng:: — string utilities
    name := "x-plus";
    show(sys::out, "strng::to_upper = {strng::to_upper(name)}\n");
    show(sys::out, "strng::replace   = {strng::replace(name, \"-\", \" plus \")}\n");

    // statistics:: — data research
    dataset := [12.5, 14.1, 13.8, 15.2];
    show(sys::out, "statistics::mean = {statistics::mean(dataset)}\n");
    show(sys::out, "statistics::std  = {statistics::std(dataset):.4}\n");
    yield 0;
}
```

The key pattern: **`using = <<name.xp>>;`** to import, then library functions
are called through the **implicit namespace** `file_name::function(...)` — e.g.
`math::sqrt`, `gpio::write`, `statistics::mean`.

---

## 20. Library Resolution System (Offline-First)

When a program writes `using = <<geomplus.xp>>`, the `xpc` resolver searches for
the library in **four levels**, in order:

```
using = <<geomplus.xp>>
   │
   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1) ./libs/geomplus.xp        (this project's local library)        │
│ 2) ~/.xpc/libs/geomplus.xp   (the user's global cache)             │
│ 3) bundled stdlib            (the libs/ shipped with the toolchain)│
│ 4) GitHub raw                (X-Ples/Liblary — silent fetch,       │
│                               5s timeout, then cached)             │
└────────────────────────────────────────────────────────────────────┘
```

Design consequences:

- **Offline-first.** Levels 1–3 need no internet. The internet is only used to
  pull new public libraries (level 4, silent with a 5-second timeout).
- **Deterministic.** The local library always wins — handy for experimental overrides.
- **`xpc.lock` lockfile** (TOML format) records the version, origin, and
  **SHA-256** of every package — builds are reproducible and auditable
  (`xpc audit`).
- **`--no-cache`** forces a re-download; `xpc purge` clears the whole cache.

---

## 21. Tutorial: Building a Private Library

We will build a library called `geomplus` (vector geometry operations) and use
it from our own program. **No GitHub, no internet required.**

### Step 1 — prepare the library folder

Local libraries live in the project's `libs/` folder (resolution level 1):

```bash
cd ~/X+
mkdir -p libs
```

### Step 2 — write the library (`libs/geomplus.xp`)

```rust
// ==========================================
// geomplus — extra X+ geometry operations
// Implicit namespace: geomplus::
// version 0.1.0
// ==========================================

pub fn distance(a: vec2, b: vec2) -> f64 {
    d := a - b;
    return sqrt(d .x d);        // sqrt = native intrinsic; `.x` is the dot product
}

pub fn midpoint(a: vec2, b: vec2) -> vec2 {
    return (a + b) * 0.5;       // `*` broadcasts the scalar
}

pub fn lerp(a: vec2, b: vec2, t: f64) -> vec2 {
    guard t >= 0.0 && t <= 1.0 else {
        // out of range: return the nearest endpoint
        if t < 0.0 { return a; }
        return b;
    }
    return a * (1.0 - t) + b * t;
}
```

> Recommended style (copied from the stdlib): a header comment with the
> namespace name, `pub fn` for the public API, helper functions without `pub`,
> and `guard` to validate preconditions. Note: `sqrt` is a **native intrinsic**
> (SPECS.md §6) — it can be called directly without an import.

### Step 3 — use it from the main program (`main.xp`)

```rust
using = <<geomplus.xp>>

main => int32 {
    a: vec2 = [0.0, 0.0];
    b: vec2 = [3.0, 4.0];

    show(sys::out, "distance     = {geomplus::distance(a, b)}\n");     // 5.0
    show(sys::out, "midpoint     = {geomplus::midpoint(a, b)}\n");     // [1.5, 2.0]
    show(sys::out, "lerp t=0.5   = {geomplus::lerp(a, b, 0.5)}\n");
    yield 0;
}
```

### Step 4 — check and run

```bash
xpc check main.xp     # validate syntax + types (including geomplus calls)
xpc run main.xp
```

The resolver finds `geomplus` at **level 1** (`./libs/`) — done.

### Step 5 — (optional) make it global for all projects

Install it into your personal cache so it can be `using`-ed from any project on
this machine (resolution level 2):

```bash
cp libs/geomplus.xp ~/.xpc/libs/   # manually copy into the personal cache
xpc list                           # verify it is installed
xpc cache-size                     # view the cache size
```

(`xpc install <name>` is for **registry** packages — useful once your library is
registered in X-Ples/Liblary, Section 22. It also automatically writes the
`xpc.lock` lockfile containing the SHA-256.)

Now `using = <<geomplus.xp>>` works in **every** project without copying the
file again. Run `xpc audit` periodically to check for outdated/corrupt libraries.

> **Real test result** of the code above (verified with `xpc run`):
> `distance = 5.0`, `midpoint = [1.5, 2.0]`, `lerp t=0.5 = [1.5, 2.0]`.

---

## 22. Tutorial: Publishing to X-Ples/Liblary (Fork + PR)

Want your library used by **every X+ user in the world**? Collect it in the
**`github.com/X-Ples/Liblary`** repository — the official central X+ registry,
with this base URL:

```
https://raw.githubusercontent.com/X-Ples/Liblary/main/
```

Once your PR is merged, anyone who writes `using = <<geomplus.xp>>` will
automatically get your library (resolution level 4: silent fetch + cache).

### Prerequisites

- A GitHub account.
- `git` installed (on Termux: `pkg install git`).
- A library that passes `xpc check`, with an example that can be run.

### Step 1 — fork the repository

Open `https://github.com/X-Ples/Liblary` → the **Fork** button. You get a copy:
`github.com/<username>/Liblary`.

### Step 2 — clone your fork and create a branch

```bash
git clone https://github.com/<username>/Liblary.git
cd Liblary
git checkout -b lib/geomplus        # branch convention: lib/<library-name>
```

### Step 3 — add the library folder

One **folder per library** — its path is what gets registered in the registry
(see the `math` entry: `path = "math/math.xp"`):

```bash
mkdir -p geomplus
# geomplus/geomplus.xp   — library code (pub fn + namespace header comment)
# geomplus/example.xp    — a runnable example program
# geomplus/README.md     — description + usage
```

### Step 4 — register it in `registry.toml`

Open `registry.toml` at the repository root and add an entry following the
existing `[packages.<name>]` format:

```toml
[packages.geomplus]
version = "0.1.0"
author = "<your-name> (<your-email>)"
path = "geomplus/geomplus.xp"
size_kb = 0.55
date = "19/Oct/2026"
description = "vec2 geometry operations: distance, midpoint, linear interpolation"
```

### Step 5 — test exactly like any other user

Simulate a user who only has the cache — no local path:

```bash
cp geomplus/geomplus.xp ~/.xpc/libs/
cd /tmp && mkdir test && cd test
# create main.xp containing using = <<geomplus.xp>> + calls, then:
xpc run main.xp
```

Also run `geomplus/example.xp` from the repo folder. **A PR whose example fails
to run will be asked for revision.**

### Step 6 — commit and push

```bash
git add geomplus/ registry.toml
git commit -m "add geomplus v0.1.0 — vec2 geometry (distance, midpoint, lerp)"
git push origin lib/geomplus
```

### Step 7 — open a Pull Request

On GitHub: your fork → **Compare & pull request** → target `X-Ples/Liblary`
branch `main`. Fill in the PR description:

```markdown
## New library: geomplus v0.1.0

**What:** vec2 geometry operations — distance, midpoint, linear interpolation.
**Why it is useful:** uses native vec2, the `.x` dot product, and the `|>`
pipeline, so it passes shape checking at compile time.

**Tested:**
- [x] xpc check passes for all files
- [x] example.xp runs: output matches the documentation
- [x] registry.toml entry follows the [packages.<name>] format
- [x] does not modify other contributors' libraries
```

### Step 8 — review and merge

Once **merged**, `xpc search geomplus` and `xpc info geomplus` will find it in
the registry, and `xpc install geomplus` pulls it straight from
`X-Ples/Liblary`. Users who write `using = <<geomplus.xp>>` without an explicit
install are served too — the resolver fetches it automatically (level 4).

### Library conventions in X-Ples/Liblary

| Convention | Reason |
|------------|--------|
| `lowercase`, unique library name | Consistent with the `name::function` namespace |
| Header comment: name + version + "Implicit namespace:" | Makes the source easy to read |
| `pub fn` for the API, helpers without `pub` | Clear API vs. internals |
| One folder per library, path in `registry.toml` | GitHub raw auto-fetch works per file |
| Complete `[packages.<name>]` entry (version/author/path/size_kb/date/description) | Consistent with the existing registry |
| Include a runnable `example.xp` | Verified PRs |
| Do not modify other contributors' files | Focused PRs that are reviewed quickly |
| A version update = a new PR with a bumped `version` | Clear change history |

---

## 23. Testing

The X+ project has a standalone test suite — 20 tests. It lives in the source
repository and is intended for **contributors** modifying the toolchain:

```bash
python3 tests/run_tests.py
```

A healthy output:

```
....................
20/20 tests passed
```

Coverage: lexer, parser (both syntax modes), type checking (including `E204`
shape checking), CTFE, XVM execution, `.xbc` packaging, and safety features
(Result/`?`/null-safety). If you modify the toolchain, run this before
committing. Additional per-file audits available from the binary:
`xpc mem-check`, `xpc bounds-check`, `xpc lint`.

---

## 24. Project Structure

This is the layout of the **source repository** (end users only need the
prebuilt `xpc` binary from the release page):

```
X+/
├── xplus/              # toolchain source package
│   ├── __init__.py
│   ├── tokens.py       # token definitions
│   ├── lexer.py        # tokenizer + visual positions
│   ├── astnodes.py     # AST nodes
│   ├── parser.py       # recursive-descent parser (dual-syntax)
│   ├── typesys.py      # type inference + tensor shape checking
│   ├── xir.py          # XIR (SSA) + .xbc container (XBC1)
│   ├── vm.py           # XVM: checked arith, arena, MMIO/GPIO, @grad
│   ├── compiler.py     # full pipeline orchestration
│   ├── resolver.py     # 4-level library resolution + lockfile
│   ├── builtins.py     # native intrinsics (sum, length, push, ...)
│   ├── errors.py       # coded errors E1xx/E2xx/E3xx + visual rendering
│   ├── cli.py          # the 40 xpc subcommands
│   └── repl.py         # interactive REPL (JIT in RAM)
├── libs/               # standard library (written in X+)
│   ├── math.xp  strng.xp  string.xp  time.xp
│   ├── statistics.xp  gpio.xp  task.xp
│   └── registry.toml   # central X-Ples/Liblary registry map
├── examples/           # 20 example programs
│   ├── hello.xp  matmul.xp  guard_result.xp  comptime.xp
│   ├── smv_demo.xp  arena_demo.xp  pipeline.xp  rational_demo.xp
│   ├── iot_led.xp  interrupt_demo.xp  library_user.xp  ...
├── tests/
│   └── run_tests.py    # 20 standalone tests
├── bench/suite/        # benchmark collector (collect.sh) — new in 1.1.0
├── xpc                 # CLI launcher
├── SPECS.md            # formal spec: EBNF, XIR, XBC, SMV, LOCKED decisions
└── README.md           # this document
```

Prebuilt binaries of the toolchain are published at
<https://github.com/X-Ples/xpc/releases/tag/binary>.

---

## 25. Real-World Uses

| Scenario | Why X+ fits |
|----------|-------------|
| **Learning compilers & VMs** | A complete but readable pipeline: lexer → AST → typesys → XIR (SSA) → XBC; `xpc ast/ir/asm/disasm` open up every stage |
| **Small-scale math & linear algebra** | `vecN`/`mat[R,C]` primitives, `x` matmul with shape guarantees, exact `rational`, `@grad` autodiff, `\|>` pipelines |
| **Virtual embedded lab** | Type-safe `register` MMIO, `asm!`, `@interrupt(vector::...)`, `xpc simulate --board=esp32` — no hardware, no pointer casts |
| **Teaching type safety** | `Result<T,E>`, `T?`, `guard`, `?`, `match` — "if it compiles, this class of error is already handled" |
| **Scripting on weak/offline devices** | A single self-contained binary; Termux/Android and older machines; offline-first resolver + lockfile; ~1.3 MB compiler peak RAM in 1.1.0 |
| **Performance tuning** | `xpc vectorize`, `sim-profile`, `cache-fit`, ns-level `bench`, SMV telemetry (`@simd_width`, `@arena_peak`) |
| **Deterministic computation** | Checked integers + `rational` remove floating-point surprises |
| **Fast edit-compile loops** | A 1000-line module compiles in ~5 ms with `xzc` — ideal for `xpc watch` |

---

## 26. Troubleshooting & FAQ

### `xpc: command not found`

The binary is not on your `PATH`. Move it into a folder that is on your `PATH`
(for example `~/.local/bin` on Linux/macOS, or `$PREFIX/bin` on Termux), or call
it with its full path (`./xpc version`). On Windows, add the folder containing
`xpc.exe` to your `PATH` environment variable and reopen the terminal.

### `Permission denied` when running `./xpc`

The downloaded file is not marked executable:

```bash
chmod +x xpc
```

### The downloaded binary does not run on my machine

Make sure you downloaded the asset that matches your operating system **and**
CPU architecture (for example, ARM64 devices such as Raspberry Pi or most
Android phones need an ARM64 build, not x86-64). All assets are listed at
<https://github.com/X-Ples/xpc/releases/tag/binary>. Run `xpc doctor` once it
starts for a health diagnosis.

### `stdlib missing` warning

The standard library could not be found. It is normally bundled with the
toolchain; if you are working from a source checkout, make sure the `libs/`
folder exists in the project root or in `~/.xpc/libs/`. While the stdlib is
absent, all language features still work — only `using = <<...>>` imports of
stdlib modules cannot be resolved.

### Compilation fails with E220

Your `comptime` expression exceeded 1,000,000 steps. Simplify the expression or
split it into several `const`s.

### `E120: unknown SMV`

The SMV name is not registered. Check the spelling (the compiler gives a
nearest-name suggestion) and see the SMV table in Section 14 / SPECS.md §5.

### `xpc version` still shows an older version after updating

Another `xpc` earlier on your `PATH` is shadowing the new one. Run
`which xpc` (or `where xpc` on Windows) and remove or replace the old copy.

### My X+ program is slower than the C++ version

Expected on some workloads — see [Section 30](#30-performance--benchmarks-110)
for honest numbers and [Section 31](#31-important-tips) for how to speed things up.

### How do I update X+?

Run `xpc self-update`, or download the latest binary from the
[release page](https://github.com/X-Ples/xpc/releases/tag/binary) and replace the
old one.

### Does X+ need the internet?

No. The internet is only needed once to pull public libraries that are not yet
cached (and to download the binary itself). The bundled stdlib and local
libraries work fully offline; `--no-cache` forces a re-download when needed.

---

## 27. Roadmap v0.5

Per SPECS.md §8:

1. **Lower XIR → Cranelift IR** (dev loop) **& LLVM IR** (release, `-O3` + LTO).
2. **Real auto-vectorization**: loops → SIMD intrinsics per target (AVX-512/NEON/RVV).
3. **Cross-compilation**: `xpc export main.xp --target=linux-aarch64`.
4. **Bare-metal flash**: `xpc flash main.xp --target=esp32 --port=/dev/ttyUSB0`.
5. **`export "C"` ABI** + FFI header parsing.
6. **Traits + monomorphization**; **coroutines** (`yield` returns as a generator).
7. **Real threads** for `parfor` / `task::spawn` (thread-local arena).
8. **Live GitHub registry** X-Ples/Liblary with commit-hash pinning.
9. **VM hot paths** (informed by 1.1.0 benchmarks): faster string handling and
   graph/collection workloads in `xpv`.

---

## 28. Ecosystem & Contributing

- **Toolchain binaries (1.1.0):**
  [`github.com/X-Ples/xpc/releases/tag/binary`](https://github.com/X-Ples/xpc/releases/tag/binary)
  — download the prebuilt `xpc` for your platform.
- **Public library repository:**
  [`github.com/X-Ples/Liblary`](https://github.com/X-Ples/Liblary) — just
  *fork* + *PR* (full guide in Section 22).
- **Formal specification:** `SPECS.md` at the project root — EBNF, XIR
  instruction set, XBC1 format, LOCKED design decisions, and the SMV table.
- **Reporting bugs / proposing features:** open an *issue* with a minimal `.xp`
  snippet that triggers the problem + the output of `xpc doctor`.
- **Contributing examples:** `examples/` is always open — one example that
  teaches one idea is worth more than a large unreadable program.

---

## 29. What's New in 1.1.0

Version **1.1.0** is a toolchain release: the language itself stays at
**v0.1 (locked mode)**, so existing `.xp` programs keep working unchanged.

| Component | Version | Role |
|-----------|---------|------|
| `xpc` | 1.1.0 | The single CLI entry point (40 subcommands) |
| `xzc` | 1.1.0 | The compiler, written in **Zig** — `.xp` → `.xbc` |
| `xpv` | 1.1.0 | The VM, with a **Rust core** (`xplus-core 1.1.0`) — runs `.xbc` |

### Highlights

1. **Much faster compilation.** On a generated 1000-line module, `xzc`
   compiles in a mean of **5.38 ms** versus **221.63 ms** for `g++` — about
   **41x faster**.
2. **Tiny bytecode.** A hello-world `.xbc` is only **83 bytes**; the `bfs`
   workload is 5,630 bytes and `cache` is 4,623 bytes. The equivalent stripped
   native ELF for hello is 14,472 bytes (~174x larger).
3. **Very small memory footprint.** The `xzc` compiler peaks at about
   **1.3 MB** RSS (g++ peaks at ~24.7 MB for the same job); the VM running
   hello peaks at about **2.9 MB**.
4. **Near-parity on memory- and native-call-heavy work.** The `mem` and
   `cache` pair workloads run within roughly 2%–10% of native C++.
5. **Verifiable results.** Every benchmark workload prints checksums and self
   checks, so a wrong compile or wrong VM result cannot silently distort the
   numbers.
6. **Reproducible benchmark suite.** `bench/suite/collect.sh` regenerates the
   whole report (warmups excluded, mean/min/median/stddev reported).
7. **One download.** All 1.1.0 binaries live at a single release page:
   <https://github.com/X-Ples/xpc/releases/tag/binary>.

### Upgrade checklist

```bash
xpc self-update          # or re-download from the release page
xpc version              # should report 1.1.0
xpc doctor               # health check
xpc check main.xp        # re-validate your existing programs
```

---

## 30. Performance & Benchmarks (1.1.0)

All numbers below were measured on **Linux x86_64 (Intel Xeon)**, with 3 warmup
runs excluded and 15 measured runs per side (20 for the startup table). The
C++ side is a single `g++ -O2` binary; the X+ side is one `xzc` compile
followed by repeated `xpv` runs, so X+ per-run cost **includes VM startup and
module load**. Your hardware will give different absolute numbers.

### 30.1 Compile speed (generated 1000-line module)

| Tool | Mean | Min | Median | Std Dev |
|------|-----:|----:|-------:|--------:|
| `xzc` | 5.38 ms | 4.70 ms | 5.13 ms | 0.55 ms |
| `g++` | 221.63 ms | 214.19 ms | 220.89 ms | 6.30 ms |

Ratio: `g++` takes **41.2x** longer. Both sides produced agreeing outputs.

### 30.2 Peak memory (VmHWM)

| Workload | Peak RSS |
|----------|---------:|
| `xzc` compile | 1,344 KB |
| `xpv` hello | 2,944 KB |
| `xpv` bfs | 6,920 KB |
| `xpv` cache | 19,612 KB |
| `g++` compile | 25,268 KB |

### 30.3 Artifact sizes

| Artifact | Bytes |
|----------|------:|
| `.xbc` hello | 83 |
| `.xbc` bfs | 5,630 |
| `.xbc` cache | 4,623 |
| ELF hello (stripped) | 14,472 |
| ELF arith (stripped) | 14,472 |
| ELF sort (stripped) | 14,608 |

### 30.4 Run-time ratios vs native C++ (pair suite)

| Scenario | C++ mean | X+ mean | Ratio |
|----------|---------:|--------:|------:|
| `mem` | 63.41 ms | 64.54 ms | **1.02x** |
| `cache` | 33.00 ms | 36.28 ms | **1.10x** |
| `io` | 42.58 ms | 84.07 ms | 1.97x |
| `str` | 204.18 ms | 1757.81 ms | 8.61x |
| `bfs` | 26.62 ms | 376.90 ms | 14.16x |

**How to read this honestly:** X+ is at parity where work is dominated by
memory access or native calls into the Rust core (`mem`, `cache`), moderately
slower on I/O, and noticeably slower on pure interpreted compute such as graph
traversal (`bfs`) and string-heavy loops (`str`). Choose X+ for fast iteration,
safety, small artifacts and low memory — not (yet) for hot pure-compute inner
loops. The v0.5 roadmap (Cranelift/LLVM lowering) targets this gap.

### 30.5 Short-program startup (fallback timer, 20 runs)

| Program | C++ | `xpc` | `xplus` |
|---------|----:|------:|--------:|
| hello | 1.28 ms | 4.45 ms | 1.70 ms |
| arith | 2.06 ms | 9.71 ms | 6.68 ms |

> The startup table uses a date-based fallback timer, so treat sub-5 ms
> numbers as indicative rather than exact. The very short-lived `hello`
> memory sample is also the most exposed to polling races (it is retried up to
> five times until a nonzero sample lands).

---

## 31. Important Tips

### Getting started
- **Always start with `xpc check`.** It validates syntax and types instantly
  with no execution — perfect as an editor-save hook or CI step.
- **Pair `xpc watch` with 1.1.0's fast compiler** for a near-instant
  edit → compile → run loop.
- **Run `xpc doctor` first** whenever something feels off, and attach its
  output to any bug report.
- **Verify the version** with `xpc version` after every update; a stale binary
  earlier on your `PATH` is the most common cause of "it still says old version".

### Writing safer code
- **Prefer `Result<T,E>` + `?`** over sentinel values, and `T?` over "magic"
  defaults; the compiler then forces every caller to handle the failure case.
- **Use `guard` for preconditions** at the top of functions to keep the happy
  path flat and readable.
- **Keep checked integers on.** Reach for `+% -% *%` only when wraparound is
  truly the intent (hashes, checksums).
- **Use `rational` for exact math** (money-like or ratio logic) instead of
  hoping `f64` rounding stays small.
- **Remember the spacing rule:** `a .x b` is a dot product, `a.x` is field
  access. `A x B` is matmul; scalar scaling is always `*`.

### Performance & memory
- **Move constants to `comptime`.** Tables (sin, factorials) are computed by the
  compiler — 0 ms at runtime. Keep them under the 1,000,000-step limit (`E220`).
- **Use `arena { ... }` for batch work** so memory is released all at once.
  Check `@arena_peak` and run `xpc mem-check` to audit leaks.
- **Lean on native intrinsics and stdlib functions** (`sqrt`, `sum`, `math::`,
  `statistics::`) for hot paths — workloads dominated by native calls run at
  near C++ speed in 1.1.0, while long hand-written interpreted loops are the
  slowest shape.
- **Profile before optimizing:** `xpc time` (compile vs. run), `xpc bench
  --runs N`, `xpc vectorize`, `xpc sim-profile`, and `xpc cache-fit`.
- **Benchmark fairly:** include warmups, report mean *and* min/median, and
  remember X+ runs include VM startup.

### Distribution & workflow
- **Ship `.xbc`, not source,** when you want tiny artifacts: `xpc build app.xp`
  then run anywhere with `xpc app.xbc`.
- **Commit `xpc.lock`** so teammates and CI get identical, SHA-256-verified
  libraries; run `xpc audit` periodically.
- **Prefer local `./libs/` for experiments** — the resolver always picks the
  local copy first. Use `--no-cache` to force a fresh download.
- **Inspect the pipeline** when debugging: `xpc ast`, `xpc ir`, `xpc disasm`.
- **Use `xpc simulate --board=esp32`** to prototype MMIO/GPIO/interrupt code
  without any hardware.

### Contributing
- **Run `python3 tests/run_tests.py`** (20/20 expected) before every commit to
  the toolchain.
- **Publishing a library?** Include a runnable `example.xp`, a complete
  `[packages.<name>]` entry, and don't touch other contributors' files
  (see [Section 22](#22-tutorial-publishing-to-x-plesliblary-fork--pr)).
- **Report bugs with a minimal `.xp` snippet** + `xpc doctor` output.

---

Happy hacking — start with `xpc repl`, then explore `examples/` from `hello.xp`
all the way to `interrupt_demo.xp`: from `show(sys::out, ...)` to MMIO and
autodiff, everything is waiting at your fingertips.
