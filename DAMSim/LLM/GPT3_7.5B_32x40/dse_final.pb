
±
6
Add_Prev_Layer(0J  5  KM  K`  
5
LayerNorm_1(0J  5  KM  K`  
7
Q(0:*   5  KE   LM  K`  
7
K(0:*   5  KE   LM  K`  
7
V(0:*   5  KE   LM  K`  
A

MHA_GEMM_1(0B+  5  KE  KM  M` 
2
SOFTMAX(0J  5  MM  M` 
4
	DropOut_1(0J  5  MM  M` 
@

MHA_GEMM_2	(0B*  5  KE  MM  K` 
F
	PROJ_GEMM
(0:1   5  KE   LM  K`hu  K  
3
	DropOut_2(0J  5  KM  K`  
5
Add_1	(0R$  5  KM  K`  υ  K
5
LayerNorm_2
(0J  5  KM  K`  
@
FFN0(0:0  5  KE   MM  L`u      
/
GeLU(0J 5  LM  L`  
B
FFN1(0:2  5  LE   MM  K`hu  K   
3
	DropOut_3(0J  5  KM  K`  
5
Add_2(0R$  5  KM  K`  υ  K7-   0=  KE  HPZAdd_Prev_LayerbLayerNorm_1 %0=  KE  HPZLayerNorm_1bQ %0=  KE  HPZLayerNorm_1bK %0=  KE  HPZLayerNorm_1bV $0=  KE  GPZQb
MHA_GEMM_1 $0=  KE  GPZKb
MHA_GEMM_1 *0=  ME  IPZ
MHA_GEMM_1bSOFTMAX )0=  ME  IPZSOFTMAXb	DropOut_1 $	0=  KE  GP	ZVb
MHA_GEMM_2 ,	0=  ME  IP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=  KE  GPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=  KE  HPZ	PROJ_GEMMb	DropOut_2 ,-   0=  KE  HPZ	DropOut_2bAdd_1 1-   0
=  KE  HPZAdd_Prev_LayerbAdd_1 .-   0=  KE  HPZAdd_1bLayerNorm_2 --   0=  KE  HPZLayerNorm_2bFFN0 &-   0=  LE  HPZFFN0bGeLU &-   0=  LE  HPZGeLUbFFN1 +-   0=  KE  HPZFFN1b	DropOut_3 ,-   0=  KE  HPZ	DropOut_3bAdd_2 (-   0=  KE  HPZAdd_1bAdd_2 U&  -   M0=   I@ H(UΝΜΜ?browwise: ¨΅   A½   AΒTPΚPP’
 ΐΜD  @S(   @  ΐA  ?%F-τύT=EτύT=M(a&>UeZήC":   ($X`h xpx*o: