
�&
:
Add_Prev_Layer ���������(0J� �5   KM   K`�
9
LayerNorm_1 ���������(0J� �5   KM   K`�
7
Q ���������(0:�� �5   KE   KM   K`�
7
K ���������(0:�� �5   KE   KM   K`�
7
V ���������(0:�� �5   KE   KM   K`�
@

MHA_GEMM_1 ���������(0B�� �5   KE   KM   M`�
5
SOFTMAX ���������(0J� �5   MM   M`�
7
	DropOut_1 ���������(0J� �5   MM   M`�
@

MHA_GEMM_2	 ���������(0B�� �5   KE   MM   K`�
F
	PROJ_GEMM
 ���������(0:&�� �5   KE   KM   K`hu   K�
7
	DropOut_2 ���������(0J� �5   KM   K`�
9
Add_1	 ���������(0R� �5   KM   K`��   K
9
LayerNorm_2
 ���������(0J� �5   KM   K`�
?
FFN0 ���������(0:$�@� �5   KE   LM   L`u   ��
2
GeLU ���������(0J�@ �5   LM   L`�
A
FFN1 ���������(0:&��@ �5   LE   LM   K`hu   K�
7
	DropOut_3 ���������(0J� �5   KM   K`�
9
Add_2 ���������(0R� �5   KM   K`��   K
4
Loss_bwd ���������(0J� �5   KM   K`�
;
DropOut_3_bwd ���������(0J� �5   KM   K`�
>
FFN1_bwd ���������(0:�@� �5   KE   LM   L`�
6
GeLU_bwd ���������(0J�@ �5   LM   L`�
E
FFN0_bwd ���������(0:&��@ �5   LE   LM   K`hu   K�
=
LayerNorm_2_bwd ���������(0J� �5   KM   K`�
;
DropOut_2_bwd ���������(0J� �5   KM   K`�
C
PROJ_GEMM_bwd ���������(0:�� �5   KE   KM   K`�
E
MHA_GEMM_2_bwd1 ���������(0B�� �5   KE   KM   M`�
E
MHA_GEMM_2_bwd2 ���������(0B�� �5   KE   MM   K`�
B
V_bwd	 ���������(0:&�� �5   KE   KM   K`hu   K�
;
DropOut_1_bwd	 ���������(0J� �5   MM   M`�
9
SOFTMAX_bwd
 ���������(0J� �5   MM   M`�
E
MHA_GEMM_1_bwd1  ���������(0B�� �5   ME   KM   K`�
E
MHA_GEMM_1_bwd2! ���������(0B�� �5   ME   KM   K`�
;
Q_bwd" ���������(0:�� �5   KE   KM   K`�
;
K_bwd# ���������(0:�� �5   KE   KM   K`�
S
FFN1_bwd_weight_update$ ���������(0B&�@� �5   KE   LM   L`hu   L�
S
FFN0_bwd_weight_update% ���������(0B&�� �@5   LE   KM   L`hu   L�
X
PROJ_GEMM_bwd_weight_update& ���������(0B&�� �5   KE   KM   K`hu   K�
P
V_bwd_weight_update'	 ���������(0B&�� �5   KE   KM   K`hu   K�
P
K_bwd_weight_update( ���������(0B&�� �5   KE   KM   K`hu   K�
P
Q_bwd_weight_update) ���������(0B&�� �5   KE   KM   K`hu   K�-0=   KPZAdd_Prev_LayerbLayerNorm_1� 0=   KPZLayerNorm_1bQ� 0=   KPZLayerNorm_1bK� 0=   KPZLayerNorm_1bV�0=   KPZQb
MHA_GEMM_1�0=   KPZKb
MHA_GEMM_1�%0=   MPZ
MHA_GEMM_1bSOFTMAX�$0=   MPZSOFTMAXb	DropOut_1�	0=   KP	ZVb
MHA_GEMM_2�'	0=   MP
Z	DropOut_1b
MHA_GEMM_2�'	
0=   KPZ
MHA_GEMM_2b	PROJ_GEMM�&
0=   KPZ	PROJ_GEMMb	DropOut_2�"0=   KPZ	DropOut_2bAdd_1�'0
=   KPZAdd_Prev_LayerbAdd_1�$0=   KPZAdd_1bLayerNorm_2�#0=   KPZLayerNorm_2bFFN0�0=   LPZFFN0bGeLU�0=   LPZGeLUbFFN1�!0=   KPZFFN1b	DropOut_3�"0=   KPZ	DropOut_3bAdd_2�0=   KPZAdd_1bAdd_2�)0=   KPZLoss_bwdbDropOut_3_bwd�)0=   KPZDropOut_3_bwdbFFN1_bwd�$0=   LPZFFN1_bwdbGeLU_bwd�$0=   LPZGeLU_bwdbFFN0_bwd�+0=   KPZFFN0_bwdbLayerNorm_2_bwd�00=   KPZLayerNorm_2_bwdbDropOut_2_bwd�.0=   KPZDropOut_2_bwdbPROJ_GEMM_bwd�00=   KPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1�$0=   KPZVbMHA_GEMM_2_bwd1�00=   KPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2�,0=   MP Z	DropOut_1bMHA_GEMM_2_bwd2�00=   MP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd�(0=   KP"ZMHA_GEMM_2_bwd2bV_bwd�,0=   MP#ZDropOut_1_bwdbSOFTMAX_bwd�. 0=   MP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1�.!0=   MP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2�$ 0
=   KP&ZKbMHA_GEMM_1_bwd1�$!0
=   KP'ZQbMHA_GEMM_1_bwd2�( "0=   KP(ZMHA_GEMM_1_bwd1bQ_bwd�(!#0=   KP)ZMHA_GEMM_1_bwd2bK_bwd�7$0=   KP*ZDropOut_3_bwdbFFN1_bwd_weight_update�.$0=   LP+ZGeLUbFFN1_bwd_weight_update�2%0	=   LP,ZGeLU_bwdbFFN0_bwd_weight_update�5%0=   KP-ZLayerNorm_2bFFN0_bwd_weight_update�<&0=   KP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update�9	&0=   KP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update�6'0=   KP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update�2'0	=   KP1ZLayerNorm_1bV_bwd_weight_update�6!(0=   KP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update�2(0=   KP3ZLayerNorm_1bK_bwd_weight_update�6 )0=   KP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update�2)0=   KP5ZLayerNorm_1bQ_bwd_weight_update�2� -  �LU�z�?"��  D�TP�
 `�D   Q-   @  �A  �?% � G-�d*<5��T=E��T=M(a&>U �;D"&:�� �(X�@`xpx��fff?�*o�:�