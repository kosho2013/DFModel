
₯
6
Add_Prev_Layer(0J 5  KM  K` 
5
LayerNorm_1(0J 5  KM  K` 
6
Q(0:)` 5  KE  "KM  K`ΐ 
6
K(0:)` 5  KE  "KM  K`ΐ 
6
V(0:)` 5  KE  "KM  K`ΐ 
>

MHA_GEMM_1(0B(` 5  KE  KM  @M``` 
1
SOFTMAX(0J 5  @MM  @M`` 
3
	DropOut_1(0J 5  @MM  @M`` 
?

MHA_GEMM_2	(0B)` 5  KE  @MM  K`ΐ 
F
	PROJ_GEMM
(0:1 5  KE  "KM  K`hu  Kΐ 
3
	DropOut_2(0J 5  KM  K` 
5
Add_1	(0R$ 5  KM  K` υ  K
5
LayerNorm_2
(0J 5  KM  K` 
?
FFN0(0:/H 5  KE  "LM  L`u    
.
GeLU(0JH 5  LM  L` 
A
FFN1(0:1H 5  LE  "LM  K`hu  K 
3
	DropOut_3(0J 5  KM  K` 
5
Add_2(0R$ 5  KM  K` υ  K7-   0=  KE  ΨGPZAdd_Prev_LayerbLayerNorm_1 %0=  KE  ΨGPZLayerNorm_1bQ %0=  KE  ΨGPZLayerNorm_1bK %0=  KE  ΨGPZLayerNorm_1bV $0=  KE  ΨFPZQb
MHA_GEMM_1 $0=  KE  ΨFPZKb
MHA_GEMM_1 *0=  @ME  IPZ
MHA_GEMM_1bSOFTMAX )0=  @ME  IPZSOFTMAXb	DropOut_1 $	0=  KE  ΨFP	ZVb
MHA_GEMM_2 ,	0=  @ME  IP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=  KE  ΨFPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=  KE  ΨGPZ	PROJ_GEMMb	DropOut_2 ,-   0=  KE  ΨGPZ	DropOut_2bAdd_1 1-   0
=  KE  ΨGPZAdd_Prev_LayerbAdd_1 .-   0=  KE  ΨGPZAdd_1bLayerNorm_2 --   0=  KE  ΨGPZLayerNorm_2bFFN0 &-   0=  LE  ΨGPZFFN0bGeLU &-   0=  LE  ΨGPZGeLUbFFN1 +-   0=  KE  ΨGPZFFN1b	DropOut_3 ,-   0=  KE  ΨGPZ	DropOut_3bAdd_2 (-   0=  KE  ΨGPZAdd_1bAdd_2 U&  -  N0=   I@(H4UΝΜΜ?browwise: ¨΅   A½   AΒTPΚPP’
 ΐΜD  @S(   @  ΐA  ?%F-τύT=EτύT=M(a&>UeZήC":` (X`h xpx*o: