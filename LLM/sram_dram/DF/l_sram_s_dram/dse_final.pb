
Ö
8
Add_Prev_Layer (0J` 5  @LM  @L``$ 
7
LayerNorm_1 (0J` 5  @LM  @L``$ 
9
Q (0:*`` 5  @LE  MM  @L``$ 
9
K (0:*`` 5  @LE  MM  @L``$ 
9
V (0:*`` 5  @LE  MM  @L``$ 
C

MHA_GEMM_1 (0B+` 5  @LE  @LM  @N`§$ 
4
SOFTMAX (0J` 5  @NM  @N`$ 
6
	DropOut_1 (0J` 5  @NM  @N`$ 
B

MHA_GEMM_2	 (0B*` 5  @LE  @NM  @L`$ 
H
	PROJ_GEMM
 (0:1`` 5  @LE  MM  @L`hu  @L`6 
5
	DropOut_2 (0J` 5  @LM  @L``$ 
7
Add_1	 (0R$` 5  @LM  @L``$ õ  @L
7
LayerNorm_2
 (0J` 5  @LM  @L``& 
B
FFN0 (0:0` 5  @LE  NM  @M`u   ``$ 
2
GeLU (0J  5  @MM  @M`Òå3 
D
FFN1 (0:2` 5  @ME  NM  @L`hu  @L``$ 
5
	DropOut_3 (0J` 5  @LM  @L``$ 
7
Add_2 (0R$` 5  @LM  @L``$ õ  @L7-   0=  @LE  XIPZAdd_Prev_LayerbLayerNorm_1 %0=  @LE  XIPZLayerNorm_1bQ %0=  @LE  XIPZLayerNorm_1bK %0=  @LE  XIPZLayerNorm_1bV $0=  @LE  XHPZQb
MHA_GEMM_1 $0=  @LE  XHPZKb
MHA_GEMM_1 *0=  @NE  XJPZ
MHA_GEMM_1bSOFTMAX )0=  @NE  XJPZSOFTMAXb	DropOut_1 $	0=  @LE  XHP	ZVb
MHA_GEMM_2 ,	0=  @NE  XJP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=  @LE  XHPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=  @LE  ¢IPZ	PROJ_GEMMb	DropOut_2 ,-   0=  @LE  XIPZ	DropOut_2bAdd_1 1-   0
=  @LE  XIPZAdd_Prev_LayerbAdd_1 .-   0=  @LE  XIPZAdd_1bLayerNorm_2 --   0=  @LE  dIPZLayerNorm_2bFFN0 &-   0=  @ME  XIPZFFN0bGeLU &-   0=  @ME°þ6JPZGeLUbFFN1 +-   0=  @LE  XIPZFFN1b	DropOut_3 ,-   0=  @LE  XIPZ	DropOut_3bAdd_2 (-   0=  @LE  XIPZAdd_1bAdd_2 @  -  úMU   ?: ¨µ  HB½  HBÂTPÊDP¢
  ÈB  S2   @  ÀA  ?%F-ôýT=5ôýT==ôýT=EôýT=M(a&>UeZÞC" :`` (`X`xpx*o:Ø