
΄
6
Add_Prev_Layer(0JP 5   LM   L`P 
5
LayerNorm_1(0JP 5   LM   L`P 
7
Q(0:*PP 5   LE  HMM   L`P 
7
K(0:*PP 5   LE  HMM   L`P 
7
V(0:*PP 5   LE  HMM   L`P 
A

MHA_GEMM_1(0B+P 5   LE   LM   N`ΐ 
2
SOFTMAX(0JP 5   NM   N`ΐ 
4
	DropOut_1(0JP 5   NM   N`ΐ 
@

MHA_GEMM_2	(0B*P 5   LE   NM   L` 
F
	PROJ_GEMM
(0:1PP 5   LE  HMM   L`hu   LP 
3
	DropOut_2(0JP 5   LM   L`P 
5
Add_1	(0R$P 5   LM   L`P υ   L
5
LayerNorm_2
(0JP 5   LM   L`P 
;
FFN0(0:+ΐP 5   LE  HNM   M`PP 
1
GeLU (0Jΐ 5   MM   M`P 
D
FFN1 (0:2Pΐ 5   ME  HNM   L`hu   LPP 
5
	DropOut_3 (0JP 5   LM   L` 
7
Add_2 (0R$P 5   LM   L` υ   L7-   0=   LE  ΄HPZAdd_Prev_LayerbLayerNorm_1 %0=   LE  ΄HPZLayerNorm_1bQ %0=   LE  ΄HPZLayerNorm_1bK %0=   LE  ΄HPZLayerNorm_1bV $0=   LE  ΄GPZQb
MHA_GEMM_1 $0=   LE  ΄GPZKb
MHA_GEMM_1 *0=   NE  ΄IPZ
MHA_GEMM_1bSOFTMAX )0=   NE  ΄IPZSOFTMAXb	DropOut_1 $	0=   LE  ΄GP	ZVb
MHA_GEMM_2 ,	0=   NE  ΄IP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=   LE  ΄GPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=   LE  ΄HPZ	PROJ_GEMMb	DropOut_2 ,-   0=   LE  ΄HPZ	DropOut_2bAdd_1 1-   0
=   LE  ΄HPZAdd_Prev_LayerbAdd_1 .-   0=   LE  ΄HPZAdd_1bLayerNorm_2 --   0=   LE  ΄HPZLayerNorm_2bFFN0 &-   0=   ME  ΄HPZFFN0bGeLU &-   0=   ME  ΘHPZGeLUbFFN1 +-   0=   LE  ΄HPZFFN1b	DropOut_3 ,-   0=   LE  ΄GPZ	DropOut_3bAdd_2 (-   0=   LE  ΄HPZAdd_1bAdd_2 X)  -  N0=   I@(H4UΝΜΜ?X browwiseZ ¨΅   A½   AΒTPΚPP’
 ΐΜD  Q(   @  ΐA  ?%F-τύT=EτύT=M(a&>UeZήC"":PP (<X`h xpx*o: