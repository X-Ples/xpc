<div align="center">

# X+ (X Plus)

**Bahasa pemrograman tensor-first yang aman: pemeriksaan bentuk matriks saat
kompilasi, null-safety absolut, arena memory, autodiff, dan dunia hardware
(MMIO/asm/interrupt) di dalam VM — toolchain 100% Python, jalan di mana saja
termasuk Android/Termux.**

`xpc toolchain` · `bahasa v0.1 (locked mode)` · `bytecode: .xbc (XBC1)` ·
`ekosistem: X-Ples/Liblary`

</div>

---

## Daftar Isi

1. [Apa itu X+?](#1-apa-itu-x)
2. [Keunggulan X+](#2-keunggulan-x)
3. [Instalasi](#3-instalasi)
4. [Program Pertama](#4-program-pertama)
5. [Tur Sintaks (Mode Locked)](#5-tur-sintaks-mode-locked)
6. [Sistem Tipe](#6-sistem-tipe)
7. [Tensor: vec, mat, `x` Matmul & `.x` Dot Product](#7-tensor-vec-mat-x-matmul--x-dot-product)
8. [Pipeline Operator `|>`](#8-pipeline-operator-)
9. [comptime — Hitung Saat Kompilasi](#9-comptime--hitung-saat-kompilasi)
10. [rational — Pecahan Eksak](#10-rational--pecahan-eksak)
11. [Keamanan: guard, Result, T?, match, dan ?](#11-keamanan-guard-result-t-match-dan-)
12. [arena — Memori Per-Scope](#12-arena--memori-per-scope)
13. [@grad & @tensor_core](#13-grad--tensor_core)
14. [SMV — System Magic Variables](#14-smv--system-magic-variables)
15. [Dunia Hardware di Dalam VM](#15-dunia-hardware-di-dalam-vm)
16. [Sistem Error & Diagnostik Visual](#16-sistem-error--diagnostik-visual)
17. [XIR & Bytecode .xbc](#17-xir--bytecode-xbc)
18. [Referensi CLI: xpc (40 Subperintah)](#18-referensi-cli-xpc-40-subperintah)
19. [Standard Library](#19-standard-library)
20. [Sistem Resolusi Library (Offline-First)](#20-sistem-resolusi-library-offline-first)
21. [Tutorial: Membuat Library Pribadi](#21-tutorial-membuat-library-pribadi)
22. [Tutorial: Publikasi ke X-Ples/Liblary (Fork + PR)](#22-tutorial-publikasi-ke-x-plesliblary-fork--pr)
23. [Pengujian](#23-pengujian)
24. [Struktur Proyek](#24-struktur-proyek)
25. [Kegunaan Nyata](#25-kegunaan-nyata)
26. [Troubleshooting & FAQ](#26-troubleshooting--faq)
27. [Roadmap v0.5](#27-roadmap-v05)
28. [Ekosistem & Kontribusi](#28-ekosistem--kontribusi)

---

## 1. Apa itu X+?

X+ (dibaca: *"X Plus"*) adalah bahasa pemrograman yang dikompilasi: kode sumber
`.xp` diolah menjadi **XIR** (*X+ Intermediate Representation*, SSA-flavored),
dikemas menjadi **bytecode `.xbc`**, lalu dieksekusi oleh **XVM** (*X+ Virtual
Machine*). Seluruh toolchain ditulis dalam Python murni — tanpa dependensi
eksternal — sehingga X+ terpasang dan berjalan di hampir semua lingkungan:
laptop, server, Raspberry Pi, bahkan ponsel Android lewat Termux.

X+ dirancang dengan tiga filosofi utama:

**Pertama, *keamanan sejak kompilasi*.** Kesalahan yang bisa dideteksi lebih awal
tidak boleh menunggu runtime. Contoh paling ikonik: mengalikan `mat[2,3] x mat[4,2]`
langsung **ditolak compiler** dengan `E204 Dimension Mismatch` — sebelum program
pertama kali dijalankan. Integer bersifat *checked-by-default* (overflow memicu
panic `E301`, bukan angka aneh diam-diam; butuh wrap eksplisit? pakai `+% -% *%`),
akses array dijaga bounds checking, dan `null` mustahil menyusup berkat tipe `T?`.

**Kedua, *matematika sebagai warga kelas satu*.** `vec2`, `mat2`, `mat[2,3]` adalah
tipe primitif bawaan; operator `x` mengalikan matriks; `.x` menghitung dot product;
`rational` berhitung pecahan secara eksak; `comptime` menghitung tabel lookup saat
kompilasi; dan `@grad` menghitung turunan fungsi secara numerik — semua tanpa
instalasi tambahan.

**Ketiga, *dunia hardware yang aman*.** X+ membawa gaya pemrograman mikrokontroler
ke dalam VM: `register GPIOA_ODR: u32 @ 0x40020014;` mendeklarasikan register MMIO
type-safe (tanpa pointer cast!), `asm! { "nop"; }` menembak instruksi level rendah,
`@interrupt(vector::EXTI0)` mendaftarkan ISR, dan `gpio::write(2, gpio::HIGH)`
menyalakan LED virtual — semua tereksekusi di atas XVM yang penuh pemeriksaan.

X+ mendukung **dua mode sintaks**: mode **legacy** (blok makro `@show(sys::out) { ... }`
dengan injeksi slot, `yield` sebagai return) yang tetap didukung penuh demi
kompatibilitas, dan mode **locked** (`show(sys::out, "...")` + `return`) yang merupakan
mode resmi. Seluruh dokumentasi ini memakai mode locked.

> **Catatan:** spesifikasi formal (EBNF, set instruksi XIR, format XBC, tabel SMV)
> ada di **`SPECS.md`**. Folder **`examples/`** berisi 20 program contoh yang bisa
> langsung dijalankan — semua cuplikan kode di dokumen ini diambil dari kode nyata
> yang lolos kompilasi.

---

## 2. Keunggulan X+

| # | Keunggulan | Penjelasan |
|---|-----------|------------|
| 1 | **Shape checking saat kompilasi** | `mat[2,3] x mat[4,2]` ditolak dengan `E204` — kesalahan aljabar linear tertangkap sebelum runtime |
| 2 | **Integer checked-by-default** | Overflow → panic `E301`; operasi wrap `+% -% *%` tersedia eksplisit |
| 3 | **Bounds checking** | Akses di luar array/tensor → `E303`, bukan memory corruption |
| 4 | **Null-safety absolut** | `T?` + `match { Some(x) =>, None => }` — tidak ada segfault kelas null |
| 5 | **`Result<T,E>` + `?`** | Propagasi error eksplisit zero-overhead, gaya Rust |
| 6 | **`guard`** | Prasyarat dini yang bersih; bahkan dipakai stdlib untuk loop max/min |
| 7 | **`comptime` (CTFE)** | Tabel lookup dihitung compiler — 0ms runtime; batas 1.000.000 langkah (`E220`) |
| 8 | **`rational`** | `1/3 + 2/7 = 13/21` eksak — tanpa galat floating-point |
| 9 | **`@grad`** | Autodiff numerik satu baris: `@grad(loss_func)` menghasilkan fungsi turunan |
| 10 | **`@tensor_core`** | Blok kernel di-offload ke jalur akselerator (SIMD, simulasi) |
| 11 | **SMV ±40 aktif** | Telemetri hardware/runtime langsung di kode: `@simd_width`, `@core_count`, `@arena_peak`, ... |
| 12 | **`arena`** | Semua alokasi blok dibersihkan serentak saat keluar scope — 0 fragmentasi |
| 13 | **Pipeline `\|>`** | `raw_signal \|> normalize \|> relu \|> scale(2.0)` — data mengalir kiri ke kanan |
| 14 | **MMIO / `asm!` / `@interrupt` / GPIO** | Lab mikrokontroler virtual yang type-safe, tanpa pointer cast |
| 15 | **Error visual berkode** | E1xx/E2xx/E3xx + W1xx warnings, dengan caret penunjuk posisi |
| 16 | **Offline-first** | Resolver 4 tingkat + lockfile `xpc.lock` (SHA-256); internet opsional |
| 17 | **Toolchain 100% Python** | Tanpa kompilasi native; Linux/macOS/Windows/Termux |
| 18 | **CLI 40 subperintah** | run/check/lint/ast/ir/disasm/build/bench/fmt/watch/install/audit/simulate/dll |
| 19 | **Ekosistem terbuka** | Library publik terpusat di `X-Ples/Liblary`; cukup fork + PR |

Singkatnya: X+ menggabungkan *type safety* ala Rust, kenyamanan scripting ala
Python, dan lab hardware ala embedded — dalam satu bahasa kecil yang konsisten.

---

## 3. Instalasi

### 3.1 Cara cepat (disarankan)

Letakkan `install.sh` di root proyek, lalu:

```bash
cd ~/X+
bash install.sh
```

Installer akan:

1. Memeriksa `python3` (3.8+; teruji hingga 3.14, termasuk Termux).
2. Memvalidasi toolchain (`import xplus.compiler, xplus.cli`) — dan **merapikan
   tata letak otomatis** bila file paket berada flat di root (penyebab error
   `ModuleNotFoundError: No module named 'xplus'`).
3. Memasang perintah global: symlink `~/.local/bin/xpc` (Linux/Mac) atau
   **`$PREFIX/bin/xpc` di Termux** — sudah otomatis ada di PATH.
4. Menjalankan smoke test (`xpc version`).
5. Idempoten: aman dijalankan berulang kali.

### 3.2 Verifikasi

```bash
xpc version                 # versi CLI + compiler + target
xpc doctor                  # diagnosis kesehatan toolchain
xpc cpu-info                # deteksi AVX-512 / NEON / FMA3
xpc run examples/hello.xp   # jalankan contoh pertama
```

### 3.3 Persyaratan

- **Python 3.8+** (disarankan 3.10+). Tidak ada dependensi pihak ketiga —
  semuanya dari standard library Python.
- Sistem operasi apa pun yang menjalankan Python, termasuk **Android/Termux**.

---

## 4. Program Pertama

Buat file `hello.xp`:

```rust
main => int32 {
    nama := "Dunia";
    speed := 999;

    show(sys::out, "Halo, {nama}! Kecepatan: {speed}\n");
    show(sys::out, "Format presisi: {speed as f64:.2}\n");

    yield 0;
}
```

Jalankan:

```bash
xpc run hello.xp
```

Output:

```
Halo, Dunia! Kecepatan: 999
Format presisi: 999.00
```

Perhatikan tiga hal khas X+:

1. **`main => int32 { ... }`** — entry point implisit (bisa ditambahi anotasi
   `@entry`; return type `void` juga sah).
2. **`show(sys::out, "...")`** — output menuju kanal `sys::out`, dengan
   interpolasi `{ekspresi}` yang dievaluasi **saat kompilasi** (zero runtime
   format error; variabel divalidasi type checker).
3. **`{speed:.2}` dan `{speed as f64}`** — format presisi dan cast eksplisit
   langsung di dalam string format.

Tiga cara mengeksekusi kode X+:

| Cara | Perintah | Kapan dipakai |
|------|----------|---------------|
| Jalankan langsung | `xpc run hello.xp` | Sehari-hari; compile ke RAM (tmpfs) + eksekusi |
| Kompilasi AOT | `xpc build hello.xp` → `xpc hello.xbc` | Distribusi bytecode; `xpc app.xbc` langsung jalan |
| REPL | `xpc repl` | Eksperimen cepat, JIT di RAM |

---

## 5. Tur Sintaks (Mode Locked)

### 5.1 Variabel: `:=` dan anotasi tipe

```rust
nama := "Dimas";            // inferensi tipe
speed: f64 = 999.0;         // dengan anotasi eksplisit
data := [1.0, 2.0, 3.0];    // array literal
nol := [0.0; 4];            // fill: [0.0, 0.0, 0.0, 0.0]
```

Dua bentuk deklarasi: `ident := expr;` (inferensi) dan `ident: type = expr;`
(eksplisit). **Statement diakhiri `;`.** Konstanta global memakai `const`
(tidak bisa dimutasi — pelanggaran ditolak compiler dengan `E208`):

```rust
const GRAVITASI: f64 = 9.80665;
const MAX_BUFFER: int32 = 4096;
```

### 5.2 Fungsi

```rust
// fungsi dengan tipe lengkap
fn bagi_cepat(a: f64, b: f64) -> f64 {
    guard b != 0.0 else {
        return 0.0;
    }
    return a / b;
}

// fungsi implisit: param tanpa tipe, return lewat yield (alias legacy)
fn normalize(data) {
    total := sum(data);
    hasil := [];
    for x in data {
        push(hasil, x / total);
    }
    yield hasil;
}
```

`return` (locked) dan `yield` (alias legacy) keduanya valid; mode locked
merekomendasikan `return` — `yield` disiapkan kembali sebagai generator di v0.5.

### 5.3 Percabangan, perulangan, match

```rust
if blink > 5 { break; }             // blok biasa
if x > 0.0 { push(hasil, x); } else { push(hasil, 0.0); }

for i in 0..4 { ... }               // range eksklusif 0,1,2,3
for i in 1..=6 { ... }              // range inklusif
loop {                              // loop tanpa kondisi + break
    blink = blink + 1;
    if blink > 5 { break; }
}

match hasil {
    Ok(v)  => { show(sys::out, "OK: {v}\n"); }
    Err(e) => { show(sys::out, "gagal: {e}\n"); }
}

match u1 {
    Some(nama) => { show(sys::out, "Ditemukan: {nama}\n"); }
    None       => { show(sys::out, "Tidak ditemukan\n"); }
}
```

### 5.4 Struct

```rust
struct Player {
    nama: string;
    speed: f64;
    level: int32;
}

main => int32 {
    p1 := Player {
        nama: "Dimas",
        speed: 999.0,
        level: 99
    };
    p1.speed = p1.speed + 1.0;      // mutasi field
    show(sys::out, "{p1.nama} speed {p1.speed}\n");
    yield 0;
}
```

### 5.5 Mode legacy (kompatibilitas)

Program lama memakai blok makro `@show` dengan injeksi slot — bahkan dengan
anotasi tipe per slot:

```rust
@show(sys::out) {
    "Player {0} punya kecepatan {1}\n";
    0::string => p1.nama;
    1::f64    => p1.speed;
};
```

Parser v0.1 menerima kedua gaya, tetapi **locked adalah mode resmi** untuk kode baru.

---

## 6. Sistem Tipe

| Keluarga | Tipe | Keterangan |
|----------|------|------------|
| Integer signed | `i8` `i16` `i32` `i64` | checked-by-default (overflow → `E301`) |
| Integer unsigned | `u8` `u16` `u32` `u64` `usize` | ideal untuk register & alamat MMIO |
| Alias umum | `int32` `int64` | pintasan yang sering dipakai di contoh |
| Float | `f32` `f64` | f32 untuk grafis/game, f64 untuk sains |
| Logika/teks | `bool` `string` `void` | |
| Eksak | `rational` | pecahan eksak; `decimal256` untuk keuangan |
| Tensor | `vec2` `vec3` `vec4`, `mat2` `mat3` `mat4`, `mat[2,3]`, `tensor[dims]` | primitif bawaan + shape checking |
| Array tetap | `[f64; 4]` | ukuran diketahui kompilasi |
| Null-safe | `T?` | `Some(v)` atau `None` |
| Error-safe | `Result<T,E>` | `Ok(v)` atau `Err(e)` |
| Buatan | `struct`, `enum`, generik `Result<T,E>` | `struct Player { ... }` |

**Prinsip penting:** `None` dan error **tidak bisa** menyusup ke tipe biasa.
Fungsi yang bisa gagal bertipe `Result<T,E>`; yang bisa kosong bertipe `T?`.
Compiler memaksa keduanya ditangani — inilah alasan program X+ nyaris bebas
crash "tak terduga".

---

## 7. Tensor: vec, mat, `x` Matmul & `.x` Dot Product

### 7.1 Membuat tensor — literal array di konteks tipe

```rust
W: mat2 = [[0.2, 0.4], [0.5, 0.8]];     // matriks 2x2
X: vec2 = [1.0, 2.0];                   // vektor 2 dimensi

A23: mat[2, 3] = [[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]];     // matriks 2x3 generik
B32: mat[3, 2] = [[7.0,  8.0],
                  [9.0,  10.0],
                  [11.0, 12.0]];        // matriks 3x2
```

Matriks dan vektor adalah **tipe primitif bawaan** — bukan library, bukan class.
Literalnya array-of-array; type checker yang memvalidasi bentuknya.

### 7.2 Perkalian matriks dengan `x` — diperiksa saat kompilasi

```rust
Y := W x X;                 // matmul mat2 x vec2 (di XIR: tensor.mul.mat2)
C := A x B;                 // matmul mat2 x mat2
C22 := A23 x B32;           // [2x3] x [3x2] -> [2x2] : LOLOS kompilasi
```

Perhatikan: kedua sisi `x` harus **tensor** (mat×mat, mat×vec, vec×mat).
Perkalian dengan skalar memakai `*` (broadcast — lihat 7.4).

Jika bentuk tidak cocok, **kompilasi gagal seketika**:

```rust
A: mat[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
B: mat[4, 2] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
C := A x B;   // <-- E204: [2x3] x [4x2]: kolom A (3) != baris B (4)
```

Pesan errornya visual dan menjelaskan solusi:

```
[X+ Type Error]: E204 Dimension Mismatch in Matrix Multiplication: [2 x 3] x [4 x 2]
 --> tensor_shape_err.xp:13:10
  |
13|     C := A x B;
  |          ^--------------------------------------
  = kolom A (3) harus sama dengan baris B (4)
```

Ini fitur paling ikonik X+: **kesalahan aljabar linear tertangkap saat kompilasi,
0ms runtime.**

### 7.3 Dot product `.x` — dan aturan spasi

| Penulisan | Arti |
|-----------|------|
| `v1 .x v2` (dipisah spasi) | **Dot product** (di XIR: `bin dot`) |
| `v.x` (menempel) | **Akses field** bernama `x` |

```rust
v1: vec4 = [1.0, 2.0, 3.0, 4.0];
v2: vec4 = [2.0, 2.0, 2.0, 2.0];

d := v1 .x v2;        // dot product -> 20.0
```

Aturan ini konsisten dan terdokumentasi di SPECS.md: **spasi = operator dot
product; menempel = member access.**

### 7.4 Broadcast skalar & elementwise

Elementwise vec (`+ - * /`) menerima skalar dan di-*broadcast* ke seluruh
elemen — **pakai `*` untuk perkalian skalar** (operator `x` ketat hanya untuk
matmul tensor × tensor):

```rust
g := v1 * 2.0;        // skalar di-broadcast ke seluruh elemen -> [2.0, 4.0, 6.0, 8.0]
h := v1 + v2;         // elementwise
d := a - b;           // elementwise (dipakai stdlib & contoh jarak)
```

### 7.5 Catatan penting token `x`

Token `x` adalah operator matmul **hanya di posisi infix** (`A x B`). Di posisi
lain ia identifier biasa — `let x = 5` gaya lama maupun `x := 5` tetap sah.

---

## 8. Pipeline Operator `|>`

Data mengalir kiri ke kanan; nilai yang di-pipe menjadi **argumen pertama** fungsi
berikutnya:

```rust
raw_signal := [0.1, -0.5, 3.2, -1.2];

raw_signal |> normalize |> relu |> scale(2.0) |> show_vec;

// Pipeline matematika klasik:
s := 30.0 |> math::deg_to_rad |> math::sin;
mag := (3.0 ^ 2.0 + 4.0 ^ 2.0) |> math::sqrt;      // |(3,4)| = 5.0
```

Pipeline membuat transformasi data terbaca seperti diagram alir — sangat pas
untuk pengolahan sinyal dan pipeline ML mini. Contoh lengkap: `examples/pipeline.xp`.

---

## 9. comptime — Hitung Saat Kompilasi

`const` + blok `comptime { ... }` dievaluasi penuh oleh compiler — hasilnya
ditanam ke bytecode, **0ms saat runtime**:

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

Aturan CTFE (sesuai keputusan LOCKED di SPECS.md):

- **Pure functions only** — tanpa I/O, network, random, atau side-effect.
- **Step limit 1.000.000 langkah** — pelanggaran menghentikan kompilasi dengan
  `E220` (bukan hang selamanya).
- Determinisme float CTFE memakai libm referensi host (v0.1); v0.5 memakai libm
  internal software-based agar build reproducible.

Kapan comptime berguna: tabel trigonometri, lookup faktorial, ekspansi ukuran
buffer/kernel, dan validasi invariant numerik lebih awal.

---

## 10. rational — Pecahan Eksak

Tipe `rational` berhitung pecahan **tanpa pembulatan** — cocok untuk kriptografi,
keuangan (`decimal256`), dan riset bilangan:

```rust
main => int32 {
    a: rational = 1/3;
    b: rational = 2/7;

    exact_sum := a + b;      // tetap pecahan eksak: 13/21
    show(sys::out, "1/3 + 2/7 = {exact_sum}\n");

    fa := 1.0 / 3.0;
    fb := 2.0 / 7.0;
    show(sys::out, "f64     : {fa + fb:.17}  (ada float error)\n");
    show(sys::out, "rational: eksak 13/21 ✓\n");
    yield 0;
}
```

Cukup anotasi `rational` — literal `1/3` di konteks tipe tersebut otomatis menjadi
pecahan eksak. Operasi `+ - * /` antar-rational menghasilkan rational kembali.

---

## 11. Keamanan: guard, Result, T?, match, dan ?

### 11.1 `guard` — prasyarat dini

`guard kondisi else { ... }` menegaskan: *"kondisi ini harus benar, kalau tidak,
keluar dari sini."* Dipakai bahkan oleh stdlib untuk mencari max/min:

```rust
fn max(data) -> f64 {
    terbesar := data[0];
    for x in data {
        guard x <= terbesar else { terbesar = x; };
    }
    yield terbesar;
}
```

### 11.2 `Result<T,E>` — sukses atau gagal, zero-overhead

```rust
fn bagi_aman(a: f64, b: f64) -> Result<f64, string> {
    guard b != 0.0 else {
        return Err("division by zero");
    }
    return Ok(a / b);
}
```

### 11.3 Operator `?` — propagasi otomatis

```rust
fn hitung_rasio(a: f64, b: f64) -> Result<f64, string> {
    hasil := bagi_aman(a, b)?;      // Err -> langsung ditularkan ke caller
    return Ok(hasil * 100.0);
}
```

### 11.4 `match` — penanganan eksplisit

```rust
r2 := bagi_aman(1.0, 0.0);
match r2 {
    Ok(v)  => { show(sys::out, "tidak mungkin\n"); }
    Err(e) => { show(sys::out, "1 / 0 -> Err({e})\n"); }
}
```

### 11.5 `T?` — optional, bebas nullptr/segfault

```rust
fn cari_user(id: int32) -> string? {
    if id == 1 { return Some("Dimas"); }
    if id == 2 { return Some("Rina"); }
    return None;
}

u9 := cari_user(9);
match u9 {
    Some(nama) => { show(sys::out, "Ditemukan: {nama}\n"); }
    None       => { show(sys::out, "ID 9: tidak ditemukan (tanpa segfault!)\n"); }
}
```

Runtime panic yang tetap terjaga meski semua di atas lolos type check:
`E301` overflow, `E302` div-zero, `E303` bounds, `E304` match fall-through,
`E305` unwrap-Err, `E306` cast overflow.

---

## 12. arena — Memori Per-Scope

`arena { ... }` adalah scope memori: seluruh alokasi di dalamnya dipantau
compiler dan **dibersihkan serentak saat blok ditutup** (RAII-style, 1 siklus,
0 fragmentasi):

```rust
main => int32 {
    arena {
        batch := buat_batch(10);
        variansi := statistics::variance(batch);
        rata := statistics::mean(batch);
        show(sys::out, "mean = {rata}\n");
    }   // <- seluruh alokasi batch lenyap bersih seketika di sini

    show(sys::out, "arena total : {@arena_total} bytes\n");
    show(sys::out, "arena peak  : {@arena_peak} bytes\n");
    show(sys::out, "stack depth : {@call_depth} frame(s)\n");
    yield 0;
}
```

Di XIR, arena diekspresikan sebagai pasangan `arena.enter` / `arena.exit` dengan
garansi cleanup. Aturan *arena-first ownership*: array `[]` dan string dialokasi
di arena scope lokal default; **global hanya boleh `const`/comptime** (global
mutabel = compile error `E208`). Telemetri memori tersedia via SMV:
`@arena_free`, `@arena_total`, `@arena_peak`, `@stack_usage`, `@heap_allocated`.

---

## 13. @grad & @tensor_core

### 13.1 `@grad(loss_func)` — autodiff numerik satu baris

`@grad` menerima nama fungsi dan menghasilkan **fungsi turunannya** (numerik):

```rust
fn loss_func(x: f64) -> f64 {
    return x ^ 2.0 + 3.0;
}

main => int32 {
    grad_loss := @grad(loss_func);
    d_val := grad_loss(4.0);
    show(sys::out, "@grad(x^2+3) di x=4.0 = {d_val:.4} (eksak: 8.0)\n");
    yield 0;
}
```

Di level XIR, fungsi ber-@grad dipanggil melalui `callv` (dynamic call). Berguna
untuk belajar kalkulus, menguji fungsi loss, dan pipeline pembelajaran mesin mini
tanpa framework eksternal.

### 13.2 `@tensor_core { ... }` — offload ke akselerator

Anotasi tensor dipakai sebagai **blok** — isi blok dieksekusi lewat jalur
akselerator (SIMD, simulasi):

```rust
@tensor_core {
    W: mat2 = [[0.2, 0.4], [0.5, 0.8]];
    X: vec2 = [1.0, 2.0];
    Y := W x X;
    show(sys::out, "tensor_core output = {Y}\n");
}
```

Keluarga blok tensor lain di spesifikasi: `@gpu`, `@deterministic`, `@resilient`,
`@enclave`. Kombinasikan dengan SMV `@simd_width` / `@tensor_backend` untuk
menyetel ukuran kernel secara dinamis (lihat `examples/vec_simd.xp`).

---

## 14. SMV — System Magic Variables

**SMV (*System Magic Variables*)** adalah variabel khusus berawalan `@` yang
disediakan lingkungan — tidak dideklarasikan, hanya-baca, dan bisa dipakai di
ekspresi mana pun (termasuk di dalam string format `show` dan konteks `comptime`).

### 14.1 Tabel SMV terimplementasi (v0.1)

| Kelompok | SMV |
|----------|-----|
| **Matematika** | `@pi` `@euler` `@tau` `@phi` |
| **Hardware/SIMD** | `@simd_width` `@simd_lane_count` `@is_simd_active` `@core_count` `@cpu_freq_mhz` `@thread_id` |
| **Telemetri** | `@time_ns` `@uptime_ms` `@random_entropy` |
| **Arena/Memori** | `@arena_free` `@arena_total` `@arena_peak` `@stack_usage` `@heap_allocated` `@is_unsafe` |
| **Build** | `@compiler_ver` `@target_arch` `@target_os` `@build_profile` `@opt_level` `@tensor_backend` `@matrix_precision` `@ctfe_active` `@git_commit` `@build_timestamp` |
| **Source/Debug** | `@file` `@line` `@col` `@function` `@call_depth` |
| **App/AI** | `@app_name` `@environment` `@timezone` `@locale` `@log_level` `@terminal_width` `@debug_mode` `@batch_size` `@epoch_index` `@learning_rate` `@loss_value` |

SMV di luar tabel (±120 lagi di spesifikasi: kelompok network, HTTP, aerospace,
quantum, dll) ditolak compiler dengan **`E120`** plus saran nama terdekat.
Roadmap v0.5 mengisinya sesuai ketersediaan backend.

### 14.2 Contoh pemakaian nyata

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

Jalankan sendiri: `xpc run examples/smv_demo.xp`. Untuk info CPU/VM yang
menyusun nilai SMV dinamis: `xpc cpu-info` (deteksi AVX-512 / NEON / FMA3).

---

## 15. Dunia Hardware di Dalam VM

Semua "hardware" hanyalah simulasi — tetapi **semantiknya serius**: alamat
register nyata, ISR dengan vector, dan instruksi CPU paling mendasar.

### 15.1 MMIO dengan `register` — type-safe, tanpa pointer cast

```rust
register GPIOA_ODR: u32 @ 0x40020014;
register GPIOA_BSRR: u32 @ 0x40020018;

main => void {
    GPIOA_ODR = 0b00000001;             // tulis register seperti variabel biasa!
    show(sys::out, "GPIOA_ODR dibaca ulang: {GPIOA_ODR}\n");

    GPIOA_ODR = GPIOA_ODR + 2;          // operasi aritmetika checked pada u32
    show(sys::out, "GPIOA_ODR + 2 = {GPIOA_ODR}\n");
    yield 0;
}
```

Deklarasinya `register NAMA: tipe @ 0xALAMAT;` — VM mensimulasikan bus dan
memvalidasi akses; tidak ada pointer mentah yang menyelinap.

### 15.2 `asm!` — inline assembly di dalam `unsafe`

```rust
unsafe {
    asm! {
        "nop";   // No Operation instruction
        "cli";   // Clear Interrupt Flag
    };
}
```

Isi `asm!` berupa statement string. Blok `unsafe` menandai area di luar jaminan
keamanan normal (lihat juga SMV `@is_unsafe`). Pasangkan dengan `xpc disasm`
untuk melihat bagaimana kode X+ Anda diturunkan menjadi instruksi.

### 15.3 `@interrupt(vector::NAMA)` — ISR registration

```rust
@interrupt(vector::EXTI0)
fn tombol_ditekan_handler() => void {
    gpio::toggle(13);
}
```

### 15.4 Simulator board

```bash
xpc run examples/iot_led.xp                          # eksekusi normal
xpc simulate examples/iot_led.xp --board=esp32       # emulator embedded on-RAM
```

Contoh `iot_led.xp` menyalakan-mematikan LED GPIO 2 (LED bawaan ESP32) via
`gpio::mode(2, gpio::OUTPUT)` → `gpio::write(2, gpio::HIGH/LOW)` →
`time::delay_ms(100)` — lab mikrokontroler tanpa satu pun kabel.

---

## 16. Sistem Error & Diagnostik Visual

Semua error X+ berkode dan berkategori:

| Kode | Kategori | Contoh |
|------|----------|--------|
| **E1xx** | Sintaks | token tak terduga, kurung tidak seimbang |
| **E120** | SMV tak dikenal | `@tidak_ada` → saran nama terdekat |
| **E2xx** | Tipe/compile | `E204` dimensi matmul, `E208` global mutabel |
| **E220** | CTFE | batas 1.000.000 langkah terlampaui |
| **E3xx** | Runtime | `E301` overflow, `E302` div-zero, `E303` bounds, `E304` match fall-through, `E305` unwrap-Err, `E306` cast overflow |
| **W1xx** | Warning | dari `xpc lint` |

Format diagnostiknya visual — baris sumber, caret penunjuk kolom, kode, dan
bantuan dalam bahasa Indonesia:

```
[X+ Type Error]: E204 Dimension Mismatch in Matrix Multiplication: [2 x 3] x [4 x 2]
 --> tensor_shape_err.xp:13:10
  |
13|     C := A x B;
  |          ^--------------------------------------
  = kolom A (3) harus sama dengan baris B (4)
```

Proteksi runtime yang bekerja otomatis: integer checked (wrap eksplisit `+% -% *%`
bila memang dimaksudkan), bounds checking, dan stack guard XVM. Error CLI keluar
dengan exit code `70` sehingga mudah ditangkap script CI.

---

## 17. XIR & Bytecode .xbc

Pipeline kompilasi X+:

```
hello.xp ─▶ lexer ─▶ parser (AST) ─▶ type checker ─▶ XIR (SSA) ─▶ .xbc ─▶ XVM
```

**XIR** (*X+ Intermediate Representation*) — SSA-flavored (temp `%N`, variabel
blok `%v_name`), tensor first-class, arena marker eksplisit, backend-agnostic
(siap diturunkan ke Cranelift/LLVM di v0.5). Kategori instruksinya:

| Kategori | Instruksi (contoh) |
|----------|--------------------|
| Dataflow | `const` `mov` `bin` `un` `cast` `smv` |
| Tensor | `bin matmul` `bin dot`, elementwise vec, broadcast skalar |
| Kontrol | `label` `jmp` `jz` `jnz` `ret` `panic` |
| Panggilan | `call @fn(args)` `callv` (dynamic: fnref/@grad) `native` |
| Output | `show chan, fmt, args, specs` |
| Memori | `array` `fill_array` `index_get` `index_set` `struct` `member` `set_member` |
| Monadic | `wrap_ok` `wrap_err` `wrap_some` `wrap_none` `propagate` (=`?`) `extract` |
| Arena | `arena.enter` `arena.exit` (RAII) |
| Konkurensi | `task.spawn` `task.wrap` `task.unwrap` (async eager) |
| Metal | `unsafe.enter/exit` `asm` `reg_read` `reg_write` `verify` |
| Trace | `line N` (source mapping untuk panic report) |

Intip sendiri:

```bash
xpc ast hello.xp       # dump AST (JSON)
xpc ir hello.xp        # dump XIR
xpc asm hello.xp       # pemetaan ilustratif ke assembly native (SIMD)
xpc disasm hello.xbc   # bongkar bytecode
```

**`.xbc`** — kontainer bytecode portabel:

```
offset  size  field
0       4     magic  "XBC1"
4       4     version u8, flags u8, reserved u16   (LE)
8       4     header_len u32 (LE) — header JSON
12      N     header JSON: xpc_version, target, fast_math, opt_level,
              build_timestamp, structs, enums, registers
12+N    4     body_len u32 (LE)
16+N    M     body: zlib(JSON { funcs, consts, structs, enums, registers })
```

File `.xbc` bisa dieksekusi langsung: `xpc app.xbc` (argumen file `.xbc`
otomatis diteruskan ke mode `run --no-artifact`).

---

## 18. Referensi CLI: xpc (40 Subperintah)

`xpc` adalah pintu masuk seluruh toolchain — **40 subperintah** dalam 7 kelompok.
Flag global: `--version`, `-s/--silent`, `--no-color`, `--target`, `--tmp-dir`,
`--no-cache`.

### Kelompok: Compile & Eksekusi

| Perintah | Fungsi |
|----------|--------|
| `xpc run <file.xp>` | Compile ke RAM (.xbc tmpfs) + eksekusi langsung |
| `xpc check <file.xp>` | Validasi sintaks + tipe (output → /dev/null, kilat) |
| `xpc lint <file.xp>` | Check + static analysis tambahan |
| `xpc ast <file.xp>` | Dump Abstract Syntax Tree (JSON) |
| `xpc ir <file.xp>` | Dump XIR (SSA-flavored, backend-agnostic) |
| `xpc asm <file.xp>` | Pemetaan ilustratif ke assembly native (SIMD) |
| `xpc disasm <file.xbc>` | Disassembly file .xbc |
| `xpc compile <file.xp>` | AOT compile → .xbc di tmpfs (bukan folder proyek) |
| `xpc build <file.xp>` | Build .xbc ke path eksplisit (distribusi) |

### Kelompok: Pengukuran

| Perintah | Fungsi |
|----------|--------|
| `xpc time <file.xp>` | Ukur waktu compile vs eksekusi |
| `xpc bench <file.xp>` | Micro-benchmark ns-level (`--runs N`) |

### Kelompok: Pengembangan

| Perintah | Fungsi |
|----------|--------|
| `xpc init` | Inisialisasi proyek X+ di direktori ini |
| `xpc new <nama>` | mkdir + init proyek baru |
| `xpc fmt <file.xp>` | Format ulang kode .xp sesuai standar |
| `xpc watch <file.xp>` | Hot-reload: recompile tiap file berubah |
| `xpc repl` | REPL interaktif X+ (JIT di RAM) |

### Kelompok: Package Manager

| Perintah | Fungsi |
|----------|--------|
| `xpc install <lib>` | Unduh library dari X-Ples/Liblary |
| `xpc uninstall <lib>` | Hapus library dari cache lokal |
| `xpc update <lib>` | Update library ke versi terbaru |
| `xpc list` | Daftar library terpasang |
| `xpc search <kata>` | Cari modul di registry |
| `xpc info <lib>` | Detail paket di registry |
| `xpc purge` | Hapus seluruh cache library |
| `xpc audit` | Audit usang/korup library lokal |

### Kelompok: Environment & Diagnosa

| Perintah | Fungsi |
|----------|--------|
| `xpc env` | Konfigurasi aktif xpc |
| `xpc doctor` | Diagnosis kesehatan toolchain |
| `xpc cpu-info` | Deteksi AVX-512 / NEON / FMA3 |
| `xpc clean` | Bersihkan tmpfs build |
| `xpc cache-size` | Ukuran cache library + tmpfs |
| `xpc flags` | Flag optimizer internal aktif |
| `xpc tune` | Tuning per mikrositus CPU (`--arch`) |
| `xpc version` | Versi CLI + compiler + target |
| `xpc self-update` | Update xpc itu sendiri |

### Kelompok: SIMD & Memori

| Perintah | Fungsi |
|----------|--------|
| `xpc vectorize <file>` | Loop mana yang bisa SIMD |
| `xpc sim-profile <file>` | Profil efisiensi vectorization |
| `xpc cache-fit <file>` | Cek muat L1 data cache |
| `xpc mem-check <file>` | Audit strict ownership + leak |
| `xpc bounds-check <file>` | Audit out-of-bounds statis |

### Kelompok: IoT

| Perintah | Fungsi |
|----------|--------|
| `xpc simulate <file>` | Simulator embedded on-RAM (`--board=esp32`) |

**Alur kerja harian yang umum:**

```bash
xpc check main.xp      # 1. validasi cepat (tanpa eksekusi)
xpc lint main.xp       # 2. static analysis tambahan
xpc run main.xp        # 3. jalankan
xpc build main.xp      # 4. produksi .xbc untuk didistribusikan
```

---

## 19. Standard Library

Standard library X+ **ditulis 100% dalam bahasa X+ sendiri** (folder `libs/`) —
kode sumbernya bisa Anda baca dan jadikan referensi gaya penulisan library.
Entri resminya terdaftar di `libs/registry.toml`:

| Library | Versi | Deskripsi resmi |
|---------|-------|-----------------|
| `math` | 2.8.1 | Fast-math trigonometri, logaritma & linear algebra (SIMD-accelerated: AVX-512 / NEON) |
| `strng` | 1.4.0 | Utilitas string: transform, query & zero-copy slice |
| `string` | 1.4.0 | Alias resmi strng.xp (kompatibilitas nama panjang) |
| `time` | 1.1.3 | Delay, timestamp & clock presisi nanodetik |
| `statistics` | 0.9.2 | Statistik dasar: mean, variance, std — vectorized loop |
| `gpio` | 0.5.0 | Bare-metal GPIO untuk ESP32/STM32 + board simulator |
| `task` | 0.3.0 | Task-based concurrency & green threads (M:N scheduler v0.5) |

Contoh konsumsi library (dari `examples/library_user.xp`):

```rust
using = <<math.xp>>
using = <<strng.xp>>
using = <<statistics.xp>>

main => int32 {
    // math:: — trigonometri SIMD-accelerated
    s := math::sin_deg(30.0);
    show(sys::out, "math::sin_deg(30) = {s:.4}\n");

    // strng:: — utilitas string
    nama := "x-plus";
    show(sys::out, "strng::to_upper = {strng::to_upper(nama)}\n");
    show(sys::out, "strng::replace   = {strng::replace(nama, \"-\", \" plus \")}\n");

    // statistics:: — riset data
    dataset := [12.5, 14.1, 13.8, 15.2];
    show(sys::out, "statistics::mean = {statistics::mean(dataset)}\n");
    show(sys::out, "statistics::std  = {statistics::std(dataset):.4}\n");
    yield 0;
}
```

Pola penting: **`using = <<nama.xp>>;`** untuk mengimpor, lalu fungsi library
dipanggil lewat **namespace implisit** `nama_file::fungsi(...)` — mis. `math::sqrt`,
`gpio::write`, `statistics::mean`.

---

## 20. Sistem Resolusi Library (Offline-First)

Ketika program menulis `using = <<geomplus.xp>>`, resolver `xpc` mencari library
dengan **urutan empat tingkat**:

```
using = <<geomplus.xp>>
   │
   ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1) ./libs/geomplus.xp        (library lokal proyek ini)            │
│ 2) ~/.xpc/libs/geomplus.xp   (cache global milik pengguna)         │
│ 3) stdlib bundled            (libs/ bawaan toolchain)              │
│ 4) GitHub raw                (X-Ples/Liblary — silent fetch,       │
│                               timeout 5s, lalu di-cache)           │
└────────────────────────────────────────────────────────────────────┘
```

Konsekuensi desain:

- **Offline-first.** Tingkat 1–3 tidak butuh internet. Internet hanya untuk
  menarik library publik baru (tingkat 4, silent dengan timeout 5 detik).
- **Deterministik.** Library lokal selalu menang — praktis untuk override percobaan.
- **Lockfile `xpc.lock`** (format TOML) mencatat version, origin, dan **SHA-256**
  tiap paket — build dapat direproduksi dan ter-audit (`xpc audit`).
- **`--no-cache`** memaksa re-download; `xpc purge` membersihkan seluruh cache.

---

## 21. Tutorial: Membuat Library Pribadi

Kita akan membuat library `geomplus` (operasi geometri vektor) dan memakainya
dari program sendiri. **Tidak perlu GitHub, tidak perlu internet.**

### Langkah 1 — siapkan folder library

Library lokal diletakkan di folder `libs/` proyek (tingkat resolusi 1):

```bash
cd ~/X+
mkdir -p libs
```

### Langkah 2 — tulis library (`libs/geomplus.xp`)

```rust
// ==========================================
// geomplus — operasi geometri tambahan X+
// Namespace implisit: geomplus::
// versi 0.1.0
// ==========================================

pub fn jarak(a: vec2, b: vec2) -> f64 {
    d := a - b;
    return sqrt(d .x d);        // sqrt = native intrinsic; `.x` dot product
}

pub fn titik_tengah(a: vec2, b: vec2) -> vec2 {
    return (a + b) * 0.5;       // `*` broadcast skalar
}

pub fn lerp(a: vec2, b: vec2, t: f64) -> vec2 {
    guard t >= 0.0 && t <= 1.0 else {
        // di luar rentang: kembalikan endpoint terdekat
        if t < 0.0 { return a; }
        return b;
    }
    return a * (1.0 - t) + b * t;
}
```

> Gaya yang disarankan (ditiru dari stdlib): header komentar dengan nama
> namespace, `pub fn` untuk API publik, fungsi helper tanpa `pub`, dan `guard`
> untuk memvalidasi prasyarat. Catatan: `sqrt` adalah **native intrinsic**
> (SPECS.md §6) — bisa dipanggil langsung tanpa import.

### Langkah 3 — pakai dari program utama (`main.xp`)

```rust
using = <<geomplus.xp>>

main => int32 {
    a: vec2 = [0.0, 0.0];
    b: vec2 = [3.0, 4.0];

    show(sys::out, "jarak        = {geomplus::jarak(a, b)}\n");        // 5.0
    show(sys::out, "titik_tengah = {geomplus::titik_tengah(a, b)}\n");  // [1.5, 2.0]
    show(sys::out, "lerp t=0.5   = {geomplus::lerp(a, b, 0.5)}\n");
    yield 0;
}
```

### Langkah 4 — periksa dan jalankan

```bash
xpc check main.xp     # validasi sintaks + tipe (termasuk panggilan geomplus)
xpc run main.xp
```

Resolver menemukan `geomplus` di **tingkat 1** (`./libs/`) — selesai.

### Langkah 5 — (opsional) jadikan global untuk semua proyek

Pasang ke cache personal agar bisa di-`using` dari proyek mana pun di mesin ini
(tingkat resolusi 2):

```bash
cp libs/geomplus.xp ~/.xpc/libs/   # salin manual ke cache personal
xpc list                           # verifikasi terpasang
xpc cache-size                     # lihat ukuran cache
```

(`xpc install <nama>` dipakai untuk paket **registry** — berguna setelah
library Anda terdaftar di X-Ples/Liblary, bagian 22. Ia juga otomatis menulis
lockfile `xpc.lock` berisi SHA-256.)

Sekarang `using = <<geomplus.xp>>` bekerja di **semua** proyek tanpa menyalin
file lagi. Gunakan `xpc audit` berkala untuk memeriksa library usang/korup.

> **Hasil uji nyata** dari kode di atas (diverifikasi dengan `xpc run`):
> `jarak = 5.0`, `titik_tengah = [1.5, 2.0]`, `lerp t=0.5 = [1.5, 2.0]`.

---

## 22. Tutorial: Publikasi ke X-Ples/Liblary (Fork + PR)

Ingin library Anda dipakai **semua pengguna X+ di dunia**? Kumpulkan di
repositori **`github.com/X-Ples/Liblary`** — pusat registry resmi X+ dengan
base URL:

```
https://raw.githubusercontent.com/X-Ples/Liblary/main/
```

Setelah PR Anda di-merge, siapa pun yang menulis `using = <<geomplus.xp>>`
otomatis mendapatkan library Anda (tingkat resolusi 4: silent fetch + cache).

### Prasyarat

- Akun GitHub.
- `git` terpasang (di Termux: `pkg install git`).
- Library lolos `xpc check` dan contohnya bisa dijalankan.

### Langkah 1 — fork repositori

Buka `https://github.com/X-Ples/Liblary` → tombol **Fork**. Anda mendapat
salinan: `github.com/<username>/Liblary`.

### Langkah 2 — clone fork dan buat branch

```bash
git clone https://github.com/<username>/Liblary.git
cd Liblary
git checkout -b lib/geomplus        # konvensi branch: lib/<nama-library>
```

### Langkah 3 — tambahkan folder library

Satu **folder per library** — path-nya yang didaftarkan di registry (lihat
entri `math`: `path = "math/math.xp"`):

```bash
mkdir -p geomplus
# geomplus/geomplus.xp   — kode library (pub fn + header komentar namespace)
# geomplus/contoh.xp     — program contoh yang bisa dijalankan
# geomplus/README.md     — deskripsi + cara pakai
```

### Langkah 4 — daftarkan di `registry.toml`

Buka `registry.toml` di root repositori, tambahkan entri mengikuti format
`[packages.<nama>]` yang sudah ada:

```toml
[packages.geomplus]
version = "0.1.0"
author = "<nama-anda> (<email-anda>)"
path = "geomplus/geomplus.xp"
size_kb = 0.55
date = "19/okt/2026"
description = "Operasi geometri vec2: jarak, titik tengah, interpolasi linear"
```

### Langkah 5 — uji persis seperti pengguna lain

Simulasikan pengguna yang hanya punya cache — tanpa jalur lokal:

```bash
cp geomplus/geomplus.xp ~/.xpc/libs/
cd /tmp && mkdir uji && cd uji
# buat main.xp berisi using = <<geomplus.xp>> + pemanggilan, lalu:
xpc run main.xp
```

Jalankan juga `geomplus/contoh.xp` dari folder repo. **PR dengan contoh yang
gagal dijalankan akan diminta revisi.**

### Langkah 6 — commit dan push

```bash
git add geomplus/ registry.toml
git commit -m "add geomplus v0.1.0 — geometri vec2 (jarak, midpoint, lerp)"
git push origin lib/geomplus
```

### Langkah 7 — buka Pull Request

Di GitHub: fork Anda → **Compare & pull request** → target `X-Ples/Liblary`
branch `main`. Isi deskripsi PR:

```markdown
## Library baru: geomplus v0.1.0

**Apa:** operasi geometri vec2 — jarak, titik tengah, interpolasi linear.
**Kenapa berguna:** memakai vec2 native, dot product `.x`, dan pipeline `|>`
sehingga lolos shape checking saat kompilasi.

**Sudah diuji:**
- [x] xpc check lolos untuk semua file
- [x] contoh.xp berjalan: output sesuai dokumentasi
- [x] entri registry.toml mengikuti format [packages.<nama>]
- [x] tidak mengubah library milik kontributor lain
```

### Langkah 8 — review dan merge

Setelah **di-merge**, `xpc search geomplus` dan `xpc info geomplus` akan
menemukannya di registry, dan `xpc install geomplus` menariknya langsung dari
`X-Ples/Liblary`. Pengguna yang menulis `using = <<geomplus.xp>>` tanpa instalasi
eksplisit pun tetap dilayani — resolver menarik otomatis (tingkat 4).

### Konvensi library di X-Ples/Liblary

| Konvensi | Alasan |
|----------|--------|
| Nama library `lowercase`, unik | Konsisten dengan namespace `nama::fungsi` |
| Header komentar: nama + versi + "Namespace implisit:" | Memudahkan pembaca sumber |
| `pub fn` untuk API, helper tanpa `pub` | API jelas vs internal |
| Satu folder per library, path di `registry.toml` | Auto-fetch GitHub raw bekerja per-file |
| Entri `[packages.<nama>]` lengkap (version/author/path/size_kb/date/description) | Konsisten dengan registry yang ada |
| Sertakan `contoh.xp` yang bisa dijalankan | PR terverifikasi |
| Jangan mengubah file milik kontributor lain | PR fokus dan cepat direview |
| Pembaruan versi = PR baru dengan `version` naik | Riwayat perubahan jelas |

---

## 23. Pengujian

Proyek X+ punya rangkaian uji mandiri — 20 tes:

```bash
python3 tests/run_tests.py
```

Output yang sehat:

```
....................
20/20 tests passed
```

Cakupannya: lexer, parser (kedua mode sintaks), type checking (termasuk
`E204` shape checking), CTFE, eksekusi XVM, paket `.xbc`, dan fitur keamanan
(Result/`?`/null-safety). Jika Anda memodifikasi toolchain, jalankan ini sebelum
commit. Audit tambahan per file: `xpc mem-check`, `xpc bounds-check`, `xpc lint`.

---

## 24. Struktur Proyek

```
X+/
├── xplus/              # paket Python toolchain
│   ├── __init__.py
│   ├── tokens.py       # definisi token
│   ├── lexer.py        # tokenizer + posisi visual
│   ├── astnodes.py     # node AST
│   ├── parser.py       # parser rekursif-turun (dual-syntax)
│   ├── typesys.py      # inferensi tipe + shape checking tensor
│   ├── xir.py          # XIR (SSA) + kontainer .xbc (XBC1)
│   ├── vm.py           # XVM: checked arith, arena, MMIO/GPIO, @grad
│   ├── compiler.py     # orkestrasi pipeline penuh
│   ├── resolver.py     # resolusi library 4 tingkat + lockfile
│   ├── builtins.py     # native intrinsics (sum, length, push, ...)
│   ├── errors.py       # error berkode E1xx/E2xx/E3xx + render visual
│   ├── cli.py          # 40 subperintah xpc
│   └── repl.py         # REPL interaktif (JIT di RAM)
├── libs/               # standard library (ditulis dalam X+)
│   ├── math.xp  strng.xp  string.xp  time.xp
│   ├── statistics.xp  gpio.xp  task.xp
│   └── registry.toml   # peta registry pusat X-Ples/Liblary
├── examples/           # 20 contoh program
│   ├── hello.xp  matmul.xp  guard_result.xp  comptime.xp
│   ├── smv_demo.xp  arena_demo.xp  pipeline.xp  rational_demo.xp
│   ├── iot_led.xp  interrupt_demo.xp  library_user.xp  ...
├── tests/
│   └── run_tests.py    # 20 tes mandiri
├── xpc                 # launcher CLI
├── install.sh          # installer (layout-adaptive + Termux)
├── SPECS.md            # spesifikasi formal: EBNF, XIR, XBC, SMV, keputusan LOCKED
└── README.md           # dokumen ini
```

---

## 25. Kegunaan Nyata

| Skenario | Kenapa X+ cocok |
|----------|-----------------|
| **Belajar compiler & VM** | Pipeline lengkap tapi terbaca: lexer → AST → typesys → XIR (SSA) → XBC; `xpc ast/ir/asm/disasm` membuka setiap tahap |
| **Matematika & aljabar linear kecil** | `vecN`/`mat[R,C]` primitif, `x` matmul dengan jaminan bentuk, `rational` eksak, `@grad` autodiff, pipeline `|>` |
| **Lab embedded virtual** | `register` MMIO type-safe, `asm!`, `@interrupt(vector::...)`, `xpc simulate --board=esp32` — tanpa hardware, tanpa pointer cast |
| **Pengajaran keamanan tipe** | `Result<T,E>`, `T?`, `guard`, `?`, `match` — "jika kompilasi lolos, kelas error ini sudah tertangani" |
| **Scripting di perangkat lemah/offline** | 100% Python, tanpa dependensi; Termux/Android dan mesin tua; resolver offline-first + lockfile |
| **Tuning performa** | `xpc vectorize`, `sim-profile`, `cache-fit`, `bench` ns-level, SMV telemetri (`@simd_width`, `@arena_peak`) |
| **Komputasi deterministik** | Integer checked + `rational` menghapus kejutan floating-point |

---

## 26. Troubleshooting & FAQ

### `ModuleNotFoundError: No module named 'xplus'`

Tata letak folder tidak sesuai: file `.py` paket berada **flat di root** (atau
folder di-rename, mis. `~/X+`). Installer lama mengasumsikan folder proyek
berisi subfolder `xplus/`. **Solusi:** jalankan `install.sh` terbaru — ia
merapikan layout otomatis. Perbaikan manual:

```bash
cd ~/X+ && mkdir -p xplus && \
for f in __init__.py astnodes.py builtins.py cli.py compiler.py errors.py \
         lexer.py parser.py repl.py resolver.py tokens.py typesys.py \
         vm.py xir.py; do mv "$f" xplus/; done && \
PYTHONPATH="$HOME/X+" python3 -c "import xplus.compiler, xplus.cli; print('OK')"
```

(`xpc.py` sengaja tetap di root — ia adalah launcher yang menambahkan root
proyek ke `sys.path` sebelum memanggil `xplus.cli`.)

### Termux: `chmod: cannot access '.../xpc'`

Salinan proyek Anda tidak memiliki file launcher `xpc` (hanya `xpc.py`).
Installer terbaru membuat launcher-nya secara otomatis dan memasangnya ke
`$PREFIX/bin/xpc` (sudah di PATH Termux). Tanpa installer:

```bash
python3 xpc.py version    # jalankan lewat xpc.py
```

### Peringatan `stdlib missing` saat install

Folder `libs/` tidak ada di salinan proyek. Solusi: unduh ulang proyek lengkap,
atau letakkan `libs/*.xp` di root proyek / `~/.xpc/libs/`. Selama stdlib belum
ada, seluruh fitur bahasa tetap berjalan — hanya `using = <<...>>` ke stdlib
yang belum bisa.

### Kompilasi gagal dengan E220

Ekspresi `comptime` Anda melebihi 1.000.000 langkah. Sederhanakan ekspresi atau
pecah menjadi beberapa `const`.

### `E120: unknown SMV`

Nama SMV tidak terdaftar. Periksa ejaan (compiler memberi saran nama terdekat)
dan lihat tabel SMV di bagian 14 / SPECS.md §5.

### Berapa versi Python minimal?

Python 3.8+ (disarankan 3.10+, teruji hingga 3.14). Tidak ada dependensi
pihak ketiga.

### Apakah X+ butuh internet?

Tidak. Internet hanya diperlukan sekali untuk menarik library publik yang belum
ter-cache. Stdlib bundled dan library lokal bekerja penuh offline; `--no-cache`
memaksa unduh ulang bila diperlukan.

---

## 27. Roadmap v0.5

Sesuai SPECS.md §8:

1. **Lower XIR → Cranelift IR** (dev loop) **& LLVM IR** (release, `-O3` + LTO).
2. **Auto-vectorize nyata**: loop → SIMD intrinsics per target (AVX-512/NEON/RVV).
3. **Cross-compilation**: `xpc export main.xp --target=linux-aarch64`.
4. **Bare-metal flash**: `xpc flash main.xp --target=esp32 --port=/dev/ttyUSB0`.
5. **`export "C"` ABI** + FFI header parsing.
6. **Traits + monomorphization**; **coroutine** (`yield` kembali sebagai generator).
7. **Threads nyata** untuk `parfor` / `task::spawn` (thread-local arena).
8. **Registry live GitHub** X-Ples/Liblary dengan commit-hash pinning.

---

## 28. Ekosistem & Kontribusi

- **Repositori library publik:** [`github.com/X-Ples/Liblary`](https://github.com/X-Ples/Liblary)
  — cukup *fork* + *PR* (panduan lengkap di bagian 22).
- **Spesifikasi formal:** `SPECS.md` di root proyek — EBNF, set instruksi XIR,
  format XBC1, keputusan desain LOCKED, dan tabel SMV.
- **Melaporkan bug / mengusulkan fitur:** buka *issue* dengan potongan `.xp`
  minimal yang memicu masalah + output `xpc doctor`.
- **Menyumbang contoh:** `examples/` selalu terbuka — satu contoh yang
  mengajarkan satu ide lebih berharga daripada program besar yang tak terbaca.

Selamat mencoba — mulai dari `xpc repl`, lalu telusuri `examples/` dari
`hello.xp` sampai `interrupt_demo.xp`: dari `show(sys::out, ...)` sampai
MMIO dan autodiff, semuanya sudah menunggu di tangan Anda.


