
ç
9
Add_Prev_Layer(0J!`  5 `jMM `jM``  
8
LayerNorm_1(0J!`  5 `jMM `jM``  
:
Q(0:-``  5 `jME  MM `jM``  
:
K(0:-``  5 `jME  MM `jM``  
:
V(0:-``  5 `jME  MM `jM``  
E

MHA_GEMM_1(0B/`   5 `jME `jMMQ`â	  
6
SOFTMAX(0J#`   5QMQ`â	  
8
	DropOut_1(0J#`   5QMQ`â	  
E

MHA_GEMM_2	(0B/`   5 `jMEQM `jM`   
I
	PROJ_GEMM
(0:4``  5 `jME  MM `jM`hu `jM`  
6
	DropOut_2(0J!`  5 `jMM `jM``  
8
Add_1	(0R'`  5 `jMM `jM``  õ `jM
8
LayerNorm_2
(0J!`  5 `jMM `jM``  
>
FFN0(0:.`  5 `jME  NM `jN` `  
2
GeLU(0J"  5 `jNM `jN`   
E
FFN1(0:5`  5 `jNE  NM `jM`hu `jM`   
6
	DropOut_3(0J!`  5 `jMM `jM``  
8
Add_2(0R'`  5 `jMM `jM``  õ `jM7-   0= `jME `jMPZAdd_Prev_LayerbLayerNorm_1 %0= `jME `jMPZLayerNorm_1bQ %0= `jME `jMPZLayerNorm_1bK %0= `jME `jMPZLayerNorm_1bV $0= `jME @KPZQb
MHA_GEMM_1 $0= `jME @KPZKb
MHA_GEMM_1 *0=QE ¼>OPZ
MHA_GEMM_1bSOFTMAX )0=QE ¼>OPZSOFTMAXb	DropOut_1 $	0= `jME @KP	ZVb
MHA_GEMM_2 ,	0=QE ¼>OP
Z	DropOut_1b
MHA_GEMM_2 ,	
0= `jME @KPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0= `jME `jMPZ	PROJ_GEMMb	DropOut_2 ,-   0= `jME `jMPZ	DropOut_2bAdd_1 1-   0
= `jME `jMPZAdd_Prev_LayerbAdd_1 .-   0= `jME `jMPZAdd_1bLayerNorm_2 --   0= `jME `jMPZLayerNorm_2bFFN0 &-   0= `jNE @LPZFFN0bGeLU &-   0= `jNE @LPZGeLUbFFN1 +-   0= `jME `jMPZFFN1b	DropOut_3 ,-   0= `jME `jMPZ	DropOut_3bAdd_2 (-   0= `jME `jMPZAdd_1bAdd_2 Fµ -  ÈQU  ?½E  ÈB: ¨µ  C½  CÂTPÊPP¢
  zE   Q(   @  ÀA  ?% è G-Ãd*<EôýT=M(a&>U ;D"-:``  (`X`h xpx  ? *o:´