
Ó
6
Add_Prev_Layer(0J` 5  @LM  @L``r 
7
LayerNorm_1 (0J` 5  @LM  @L``r 
9
Q (0:*`` 5  @LE  MM  @L``r 
9
K (0:*`` 5  @LE  MM  @L``r 
9
V (0:*`` 5  @LE  MM  @L``r 
C

MHA_GEMM_1 (0B+` 5  @LE  @LM  @N`r 
4
SOFTMAX (0J` 5  @NM  @N`r 
6
	DropOut_1 (0J` 5  @NM  @N`r 
B

MHA_GEMM_2	 (0B*` 5  @LE  @NM  @L`¼&r 
H
	PROJ_GEMM
 	(0:1`` 5  @LE  MM  @L`hu  @L`©r 
5
	DropOut_2 
(0J` 5  @LM  @L``r 
7
Add_1	 (0R$` 5  @LM  @L``r õ  @L
7
LayerNorm_2
 (0J` 5  @LM  @L``r 
B
FFN0 (0:0` 5  @LE  NM  @M`u   ``r 
1
GeLU (0J 5  @MM  @M``r 
D
FFN1 (0:2` 5  @ME  NM  @L`hu  @L``r 
5
	DropOut_3 (0J` 5  @LM  @L``r 
7
Add_2 (0R$` 5  @LM  @L``r õ  @L7-   0=  @LE  +JPZAdd_Prev_LayerbLayerNorm_1 %0=  @LE  +JPZLayerNorm_1bQ %0=  @LE  +JPZLayerNorm_1bK %0=  @LE  +JPZLayerNorm_1bV $0=  @LE  +IPZQb
MHA_GEMM_1 $0=  @LE  +IPZKb
MHA_GEMM_1 *0=  @NE  +KPZ
MHA_GEMM_1bSOFTMAX )0=  @NE  +KPZSOFTMAXb	DropOut_1 $	0=  @LE  +IP	ZVb
MHA_GEMM_2 ,	0=  @NE  +KP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=  @LE  +IPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=  @LE  +JPZ	PROJ_GEMMb	DropOut_2 ,-   0=  @LE  +JPZ	DropOut_2bAdd_1 1-   0
=  @LE  +JPZAdd_Prev_LayerbAdd_1 .-   0=  @LE  +JPZAdd_1bLayerNorm_2 --   0=  @LE  +JPZLayerNorm_2bFFN0 &-   0=  @ME  +JPZFFN0bGeLU &-   0=  @ME  +JPZGeLUbFFN1 +-   0=  @LE  +JPZFFN1b	DropOut_3 ,-   0=  @LE  +JPZ	DropOut_3bAdd_2 (-   0=  @LE  +JPZAdd_1bAdd_2 @  -  MU   ?: ¨µ  HB½  HBÂTPÊDP¢
  ÈB  S2   @  ÀA  ?%F-ôýT=5ôýT==ôýT=EôýT=M(a&>UeZÞC":`` (`X`xp*o:´