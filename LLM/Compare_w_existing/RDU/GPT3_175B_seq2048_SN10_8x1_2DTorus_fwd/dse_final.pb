
ć
Y
Add_Prev_LayerE` (5  @L=  żE  żM  @LP˙˙˙˙˙˙˙˙˙X˙˙˙˙˙˙˙˙˙`` 
O
LayerNorm_1<` (5  @L=  żE  żM  @LPX˙˙˙˙˙˙˙˙˙` (
Q
QH`` (5  @L=  żE  MM  @LPX˙˙˙˙˙˙˙˙˙`u   }  L` (
Q
KH`` (5  @L=  żE  MM  @LPX˙˙˙˙˙˙˙˙˙`u   }  L` (
Q
VH`` (5  @L=  żE  MM  @LPX˙˙˙˙˙˙˙˙˙`u   }  L` (
M

MHA_GEMM_1;` (5  @L=  @LE  żM  @NPX`u   Ŕ (
L
SOFTMAX=` (5  @N=  żE  żM  @NPX˙˙˙˙˙˙˙˙˙`Ŕ (
N
	DropOut_1=` (5  @N=  żE  żM  @NPX˙˙˙˙˙˙˙˙˙`Ŕ (
L

MHA_GEMM_2	:` (5  @L=  @NE  żM  @LPX`u    (
[
	PROJ_GEMM
J`` (5  @L=  żE  MM  @LP	X˙˙˙˙˙˙˙˙˙`hu  @L}  L` (
M
	DropOut_2<` (5  @L=  żE  żM  @LP
X˙˙˙˙˙˙˙˙˙`` (
@
Add_13` (5  @L=  @LE  żM  @LPX`` (	
O
LayerNorm_2<` (5  @L=  żE  żM  @LPX˙˙˙˙˙˙˙˙˙`` (

U
FFN0I` (5  @L=  żE  NM  @MPX˙˙˙˙˙˙˙˙˙`u   }  M0` (
K
GeLU= (5  @M=  żE  żM  @MPX˙˙˙˙˙˙˙˙˙`0 (0
Y
FFN1K` (5  @M=  żE  NM  @LPX˙˙˙˙˙˙˙˙˙`hu  @L}  M`0 (0
O
	DropOut_3<` (5  @L=  żE  żM  @LPX˙˙˙˙˙˙˙˙˙` (0
B
Add_23` (5  @L=  żE  żM  @LPX` (0/0=  @LE  @IPZAdd_Prev_LayerbLayerNorm_1"0=  @LE  ŔGPZLayerNorm_1bQ"0=  @LE  ŔGPZLayerNorm_1bK"0=  @LE  ŔGPZLayerNorm_1bV!0=  @LE  ŔGPZQb
MHA_GEMM_1!0=  @LE  ŔGPZKb
MHA_GEMM_1,-   0=  @NE  ŔIPZ
MHA_GEMM_1bSOFTMAX+-   0=  @NE  ŔIPZSOFTMAXb	DropOut_1!	0=  @LE  ŔGP	ZVb
MHA_GEMM_2)	0=  @NE  ŔIP
Z	DropOut_1b
MHA_GEMM_2)	
0=  @LE  ŔGPZ
MHA_GEMM_2b	PROJ_GEMM(
0=  @LE  @IPZ	PROJ_GEMMb	DropOut_2$0=  @LE  @IPZ	DropOut_2bAdd_1)0
=  @LE  @IPZAdd_Prev_LayerbAdd_1&0=  @LE  @IPZAdd_1bLayerNorm_2%0=  @LE  @IPZLayerNorm_2bFFN0#-   0=  @ME  ŔHPZFFN0bGeLU0=  @ME  ŔHPZGeLUbFFN1#0=  @LE  @IPZFFN1b	DropOut_3$0=  @LE  ŔGPZ	DropOut_3bAdd_2 0=  @LE  @IPZAdd_1bAdd_2J
8_Chip_SN10_RDU%  ČA-  ČA:%
SN10_RDU  -   M5   ?=  ČBE  ŔS@H   @  ŔA  ?" (`8 P`X`` 