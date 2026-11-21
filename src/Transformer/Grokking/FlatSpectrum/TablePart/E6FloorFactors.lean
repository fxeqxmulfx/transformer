module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorDigits
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorPhase
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Mask

/-!
# PT6: the low-rank part of a completion of `⌊a b / 97⌋`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Two matrices `P` and
`Q` of 97 rows and 4 columns, in units of `2⁻¹⁰`, whose product `L = P Qᴴ` stays close to the table
on the training half. The fields are a hint. The kernel of Lean adds up the three numbers the proof
needs: the size of the factors (`factorSum_le`), the distance from `L` to the rounded table on the
seen cells (`residSum_le`), and the cost of the rounding there (`roundSum_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

/-- The factors `Re P`, `Im P`, `Re Q`, `Im Q`, in fields of 12 bits, each shifted up by 2048.
Within each, the entry `(i, t)` has index `97 t + i`. -/
def factorPack : ℕ := fromWords 368 [
  0x5ab48b2db15afaae1ac5aaaa8ca6da54a35a159f89da9bb9a09839689449299098ec8cf8b889687c86484782f832,
  0x1bd2be1beebfcc09c12c1ec26c2cc30c32c37c38c37c35c32c2cc27c1fc15c09bfebf1be2bd3bc2bb1b9eb8cb72b,
  0x8958b28d18eb90992394495f97f99d9bd9d99f6a18a35a4fa6ea8eaa7ac4adfafab11b2eb44b5ab76b86b9cbadbc,
  0x32a3da4ba58a5fa67a69a70a71a6ea6aa6da63a55a50a40a35a26a169ff9ed9d89c39a899299081582c84586187d,
  0x475177078279b7ba7cb7f27f181883584e86a88a8a78c78e18ff91a93794c9699849989a89c79de9eea03a15a21a,
  0x7e27c57b079077675f7497397217147036f86ee6e76d96d36ca6cb6cb6ce6ce6db6d76de6ea6f770271172373774,
  0x047fd8037fd7f77f37ea7e07d87bd7b479e78677275e7367146ef6c56aa6a39329098e98c38a288286083c820802,
  0x66e16e06e06e36ee6f66fa70c71671f72d73f75475976277078c7917a07ab7b37c47d67e17dc7f07f57fb7fe7fd8,
  0x8ad89488386485383981a8017e37d77b479d78c77476574b73973871d7137006f76ef6f86e26d46d96cf6ce6cd6d,
  0x1a80980780080b7f37f77fb80b7ea8067fe8088088138219a69a499398798497595f95694393a91d9058f68d68ca,
  0x27ff82082a82483f84684685484a84985185387a86885185f86d86284c83f84884c83a83784382b83382082880d8,
  0x7ba7bf7b27a078177f7a177b7997977a479778379477d7907c67a57c37aa7b07db7d07d27dc7fc8057e280f7de81,
  0xe24e34e54ea4ed4f84fe50751352b52784a83984584483b83583580c8188058067f47fa7f57eb7d57bb7bb7c57b8,
  0x06f06d36b36956736576406206065e95cf5bb5a358e57856755154453252351a50b5044fa4f24eb4ea4e74e14e14,
  0xa9ea85a76a5ea48a30a169f99e39c99ae98d96e94e9329118f38cf8ad88a8688488248027dd7bd79c77875573471,
  0x9547b465449426429acfae3af1af6b00b0cb15b19b1db1eb22b21b1eb1eb19b15b10b08afeaf3ae8ad9acdac2aaf,
  0xa7dc7ce7c17af79678477275573d72270b6ee6ce6af69167365462f6105f55cc5af58d57154d52a50c4f14d24b74,
  0x67e6a66bb6d46f370c72573b75076e7847997ae7bf7cf7dd7f17fb7fa80780d80e81481581681480e8087ff7f47e,
  0x9f3ff40340940941441b42c43744345346747748749e4b34ce4e74fd51a53555156d5935b25c85eb603622647662,
  0xe8a189188487385c84483681d8037e27d37b679077376373971c7006ea6c369d67c66865362d6135f75e15b35a25,
  0x82982d83c85085186387588088c88f8a08a58bf8c48bd8d08d38d68e28e98ea8ee8e78ec8e08df8d18ce8c78c58a,
  0x9d97994792d9148f18d68ba89688586f85d85083682781180e8068007f07f57fa7f27f67fd80780081180e81a81e,
  0x27b47997a87987937bf7c27b67b37d97d17e97e77f580181182183c85986689689f8ab89f90d905a50a27a049c89,
  0x8028167fe7fb7f27fc8298028018097ed7fd7ea7e77fe7ee7df7d57c87dd7cc7bd7b879f7a87b27ad7967a17a37b,
  0x158098117fc8007de7db7d07b87b77b07c67c87be7ba7a67bf7cc7be7d57ca7e47d47e37e07e87e17f37e37d67d5,
  0x4b10b18b1db24b2db31b34b34b37b35b31b2bb2ab21b1bb12b09afeaf1adbadc9318d58b28ce88f88386b8428288,
  0x7dd80282484586f8878ac8d28ef9149349529719939af9cd9e9a02a1ca3aa50a68a7da95aa4ab9acbadaae7af8b0,
  0x0450a51151f52753153f54e55e56d5825965a95c45d85ee60d62264165a6796956b36d46f17157357577797977be,
  0xd7987b07ce7e87ff82083b85d8828a28c08e690892a95195153652451950f50a4ff5004f74f54f64f44f64fa5005,
  0x7d07b67a378877176275273772971a7137036fa6f16f06eb6e66ec6e86e66f26f970270871772173274475276977,
  0x84a7fa72a6ba5ba5aa44a2da22a0c9f69e29c59ba99898696894d9319218fc8df8c08a488786d84a83380e80d7e9,
  0x37196fc6d76b469a66f64361a5ea5b65b29b39cf9e39f6a13a22a34a48a58a64a6fa79a81a87a8fa88a8ba8ca89a,
  0x7a37a97bb7be7dc7d27e37ea7f87fc8028077ff81281481381f80c7f78087f47f07e67d27c67ab79778877875b73,
  0xd57c17a679978277076275674e74373972a72e72a72c72471e72f72972c73a73474a74575b76876f77577f792793,
  0x97e17d37d27ce7da7e0a76a73a4ba32a119f69d69bd99798996093e92490e8e08c48a988286b85283381c7fc7e77,
  0x85685d86888687086787684b88587787685384e85084f86a85183b82983082e81681180b80980d8007e27f17d17d,
  0xa37d378979d79a79a7bd7ab79e7c57c67cc7cf7d17bb7f27fc8097fe7fc7d97fb81583f82982484483684c848848,
  0xe7ff7e17de7f67fe81c80781182482380b7f97f97f57fb7f07de7e17e77d67da7cb7c87ba7b07bb7af7907b37997,
  0x4b34cc4e54ff51653255056b5875a55c35e45fe61c63965967b6986b46d76f270e72b74c76978279c7b57cd7eb7f,
  0x354264154073fd3f03e73df3d63d13cd3ca3c93c93ca3d03d33d73df3e73f23fe40b41942943a44a45e47048549c,
  0x87507357196fb6dc6c16a768966a64e6306165f55d55b659d58056454852c5154f84dd4c44ac49748146c4594464,
  0x5f35d05be59a57d56054552e50c4f54dd4c34af49b48b47646445144743843242542141e41740c40a7ba79f78b76,
  0x1781c81c81f81e81c8188128068087fa7ef7d87cc7b67a479077a75d7477367186fc6e06c36aa68966a654630610,
  0xb53255457759a5b15da5f261e63e65c6786986b96d66f470e72a74875e77878d7a27b77c87d77ea7f380080b8118,
  0x81a83584885a87d8888a48be8d58ea8fb91092692c94495195d96c97797b97f97f42a44a46548649b4b84d54f751,
  0xec6ee6de6dd6d26d36ce6ca6c56c36c66d06d86dd6de6f46f56fe70e72072573e74a76177778c7a77b87d57e77ff,
  0x27f07fe80981780e80f8058077f97fb7ef7e57e47d07c87c37aa79e79c78677376e75374173672d72171570e7006,
  0x83d83281380d81180c7fb7e77cd7b779a7797677567416f06f46b56c66f471973c75b7767897957bc7c27d07e07f,
  0xe97e28037f780480080782180f80d7f881c81582582f82883582d82484584c86084f84c84385286384184983383d,
  0xf84984d85d85286284f86687085f86c85283785185784984d82c83382b82781180b82b81b8128047f98007ff7f67,
  0x6d072e72f76275277a78f7ab7c57eb7e47ff81982b82283283082284]

/-- A field of the factors. -/
def fac (k : ℕ) : ℤ := (((factorPack >>> (12 * k)) &&& 4095 : ℕ) : ℤ) - 2048

/-- The real part of `P`, in units of `2⁻¹⁰`. -/
def pRe (i t : ℕ) : ℤ := fac (t * 97 + i)

/-- The imaginary part of `P`, in units of `2⁻¹⁰`. -/
def pIm (i t : ℕ) : ℤ := fac (388 + (t * 97 + i))

/-- The real part of `Q`, in units of `2⁻¹⁰`. -/
def qRe (j t : ℕ) : ℤ := fac (776 + (t * 97 + j))

/-- The imaginary part of `Q`, in units of `2⁻¹⁰`. -/
def qIm (j t : ℕ) : ℤ := fac (1164 + (t * 97 + j))

/-- The real part of `L = P Qᴴ`, in units of `2⁻²⁰`. -/
def lowRe (i j : ℕ) : ℤ := sumFrom (fun t => pRe i t * qRe j t + pIm i t * qIm j t) 0 4

/-- The imaginary part of `L = P Qᴴ`, in units of `2⁻²⁰`. -/
def lowIm (i j : ℕ) : ℤ := sumFrom (fun t => pIm i t * qRe j t - pRe i t * qIm j t) 0 4

/-- The square of an integer. -/
def sqZ (x : ℤ) : ℤ := x * x

/-- The squared size of the row `i` of both factors, in units of `2⁻²⁰`. -/
def factorTerm (i t : ℕ) : ℤ := sqZ (pRe i t) + sqZ (pIm i t) + sqZ (qRe i t) + sqZ (qIm i t)

/-- **The kernel bounds the size of the factors**: `‖P‖²_F + ‖Q‖²_F ≤ 281`. -/
theorem factorSum_le :
    sumFrom (fun i => sumFrom (fun t => factorTerm i t) 0 4) 0 97 ≤ 294649856 := by
  decide +kernel

/-- The squared distance from `L` to the rounded table at a cell, if the cell is seen. -/
def residTerm (i j : ℕ) : ℤ :=
  if halfMask.testBit (97 * i + j) then
    sqZ (phC (i * j / 97) - lowRe i j) + sqZ (phS (i * j / 97) - lowIm i j)
  else 0

/-- **The kernel bounds the distance from `L` to the table on the seen cells**, in `2⁻⁴⁰`. -/
theorem residSum_le : sumFrom (fun i => sumFrom (fun j => residTerm i j) 0 97) 0 97 ≤
    1687000000000 := by
  decide +kernel

/-- The squared bound on the rounding of the table at a cell, if the cell is seen. -/
def roundTerm (i j : ℕ) : ℤ :=
  if halfMask.testBit (97 * i + j) then sqZ (4 * ((i * j / 97 : ℕ) + 1)) else 0

/-- **The kernel bounds the rounding of the table on the seen cells**, in units of `2⁻⁴⁰`. -/
theorem roundSum_le : sumFrom (fun i => sumFrom (fun j => roundTerm i j) 0 97) 0 97 ≤ 90000000 := by
  decide +kernel

end Transformer.Grokking.FlatSpectrum.TablePart.E6
