
�&
:
Add_Prev_Layer ���������(0J� �5  �JM  �J`�
9
LayerNorm_1 ���������(0J� �5  �JM  �J`�
7
Q ���������(0:�� �5  �JE   KM  �J`�
7
K ���������(0:�� �5  �JE   KM  �J`�
7
V ���������(0:�� �5  �JE   KM  �J`�
@

MHA_GEMM_1 ���������(0B�� �5  �JE  �JM   L`�
5
SOFTMAX ���������(0J� �5   LM   L`�
7
	DropOut_1 ���������(0J� �5   LM   L`�
@

MHA_GEMM_2	 ���������(0B�� �5  �JE   LM  �J`�
F
	PROJ_GEMM
 ���������(0:&�� �5  �JE   KM  �J`hu  �J�
7
	DropOut_2 ���������(0J� �5  �JM  �J`�
9
Add_1	 ���������(0R� �5  �JM  �J`��  �J
9
LayerNorm_2
 ���������(0J� �5  �JM  �J`�
:
FFN0 ���������(0:�@� �5  �JE   LM  �K`�
2
GeLU ���������(0J�@ �5  �KM  �K`�
A
FFN1 ���������(0:&��@ �5  �KE   LM  �J`hu  �J�
7
	DropOut_3 ���������(0J� �5  �JM  �J`�
9
Add_2 ���������(0R� �5  �JM  �J`��  �J
4
Loss_bwd ���������(0J� �5  �JM  �J`�
;
DropOut_3_bwd ���������(0J� �5  �JM  �J`�
C
FFN1_bwd ���������(0:$�@� �5  �JE   LM  �K`u   ��
6
GeLU_bwd ���������(0J�@ �5  �KM  �K`�
E
FFN0_bwd ���������(0:&��@ �5  �KE   LM  �J`hu  �J�
=
LayerNorm_2_bwd ���������(0J� �5  �JM  �J`�
;
DropOut_2_bwd ���������(0J� �5  �JM  �J`�
H
PROJ_GEMM_bwd ���������(0:$�� �5  �JE   KM  �J`u   ��
E
MHA_GEMM_2_bwd1 ���������(0B�� �5  �JE  �JM   L`�
E
MHA_GEMM_2_bwd2 ���������(0B�� �5  �JE   LM  �J`�
B
V_bwd	 ���������(0:&�� �5  �JE   KM  �J`hu  �J�
;
DropOut_1_bwd	 ���������(0J� �5   LM   L`�
9
SOFTMAX_bwd
 ���������(0J� �5   LM   L`�
E
MHA_GEMM_1_bwd1  ���������(0B�� �5   LE  �JM  �J`�
E
MHA_GEMM_1_bwd2! ���������(0B�� �5   LE  �JM  �J`�
;
Q_bwd" ���������(0:�� �5  �JE   KM  �J`�
;
K_bwd# ���������(0:�� �5  �JE   KM  �J`�
S
FFN1_bwd_weight_update$ ���������(0B&�@� �5  �JE  �KM   L`hu   L�
S
FFN0_bwd_weight_update% ���������(0B&�� �@5  �KE  �JM   L`hu   L�
X
PROJ_GEMM_bwd_weight_update& ���������(0B&�� �5  �JE  �JM   K`hu   K�
P
V_bwd_weight_update'	 ���������(0B&�� �5  �JE  �JM   K`hu   K�
P
K_bwd_weight_update( ���������(0B&�� �5  �JE  �JM   K`hu   K�
P
Q_bwd_weight_update) ���������(0B&�� �5  �JE  �JM   K`hu   K�-0=  �JPZAdd_Prev_LayerbLayerNorm_1� 0=  �JPZLayerNorm_1bQ� 0=  �JPZLayerNorm_1bK� 0=  �JPZLayerNorm_1bV�0=  �JPZQb
MHA_GEMM_1�0=  �JPZKb
MHA_GEMM_1�%0=   LPZ
MHA_GEMM_1bSOFTMAX�$0=   LPZSOFTMAXb	DropOut_1�	0=  �JP	ZVb
MHA_GEMM_2�'	0=   LP
Z	DropOut_1b
MHA_GEMM_2�'	
0=  �JPZ
MHA_GEMM_2b	PROJ_GEMM�&
0=  �JPZ	PROJ_GEMMb	DropOut_2�"0=  �JPZ	DropOut_2bAdd_1�'0
=  �JPZAdd_Prev_LayerbAdd_1�$0=  �JPZAdd_1bLayerNorm_2�#0=  �JPZLayerNorm_2bFFN0�0=  �KPZFFN0bGeLU�0=  �KPZGeLUbFFN1�!0=  �JPZFFN1b	DropOut_3�"0=  �JPZ	DropOut_3bAdd_2�0=  �JPZAdd_1bAdd_2�)0=  �JPZLoss_bwdbDropOut_3_bwd�)0=  �JPZDropOut_3_bwdbFFN1_bwd�$0=  �KPZFFN1_bwdbGeLU_bwd�$0=  �KPZGeLU_bwdbFFN0_bwd�+0=  �JPZFFN0_bwdbLayerNorm_2_bwd�00=  �JPZLayerNorm_2_bwdbDropOut_2_bwd�.0=  �JPZDropOut_2_bwdbPROJ_GEMM_bwd�00=  �JPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1�$0=  �JPZVbMHA_GEMM_2_bwd1�00=  �JPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2�,0=   LP Z	DropOut_1bMHA_GEMM_2_bwd2�00=   LP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd�(0=  �JP"ZMHA_GEMM_2_bwd2bV_bwd�,0=   LP#ZDropOut_1_bwdbSOFTMAX_bwd�. 0=   LP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1�.!0=   LP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2�$ 0
=  �JP&ZKbMHA_GEMM_1_bwd1�$!0
=  �JP'ZQbMHA_GEMM_1_bwd2�( "0=  �JP(ZMHA_GEMM_1_bwd1bQ_bwd�(!#0=  �JP)ZMHA_GEMM_1_bwd2bK_bwd�7$0=  �JP*ZDropOut_3_bwdbFFN1_bwd_weight_update�.$0=  �KP+ZGeLUbFFN1_bwd_weight_update�2%0	=  �KP,ZGeLU_bwdbFFN0_bwd_weight_update�5%0=  �JP-ZLayerNorm_2bFFN0_bwd_weight_update�<&0=  �JP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update�9	&0=  �JP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update�6'0=  �JP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update�2'0	=  �JP1ZLayerNorm_1bV_bwd_weight_update�6!(0=  �JP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update�2(0=  �JP3ZLayerNorm_1bK_bwd_weight_update�6 )0=  �JP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update�2)0=  �JP5ZLayerNorm_1bQ_bwd_weight_update�2� -  �LU�z�?"��  D�TP�
 `�D   Q-   @  �A  �?% � G-�d*<5��T=E��T=M(a&>U �;D"&:�� �(X�@`xpx��fff?�*o�:�