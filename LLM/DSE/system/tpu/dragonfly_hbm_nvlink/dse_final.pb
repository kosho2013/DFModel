
�)
8
Add_Prev_Layer(0J �� �5  �LM  �L`����E�
9
LayerNorm_1 (0J �� �5  �LM  �L`����E�
<
Q (0:-���� �5  �LE @�NM  �L`������E�
<
K (0:-���� �5  �LE @�NM  �L`������E�
<
V (0:-���� �5  �LE @�NM  �L`������E�
D

MHA_GEMM_1 (0B,��� �5  �LE  �LM  �N`������E�
5
SOFTMAX (0J �� �5  �NM  �N`����E�
7
	DropOut_1 (0J �� �5  �NM  �N`����E�
C

MHA_GEMM_2	 (0B+��� �5  �LE  �NM  �L`�����E�
K
	PROJ_GEMM
 	(0:4���� �5  �LE @�NM  �L`hu  �L������E�
7
	DropOut_2 
(0J �� �5  �LM  �L`����E�
9
Add_1	 (0R&�� �5  �LM  �L`����E��  �L
9
LayerNorm_2
 (0J �� �5  �LM  �L`����E�
D
FFN0 (0:2���� �5  �LE @�OM  �M`u   ���d����E�
1
GeLU (0J�� �5  �MM  �M`��d�E�
F
FFN1 (0:4���� �5  �ME @�OM  �L`hu  �L�����d�E�
7
	DropOut_3 (0J �� �5  �LM  �L`����E�
8
Add_2 (0R%�� �5  �LM  �L`���E��  �L
4
Loss_bwd (0J �� �5  �LM  �L`����E�
;
DropOut_3_bwd (0J �� �5  �LM  �L`����E�
C
FFN1_bwd (0:-���� �5  �LE @�OM  �M`��d����E�
5
GeLU_bwd (0J�� �5  �MM  �M`��d�E�
J
FFN0_bwd (0:4���� �5  �ME @�OM  �L`hu  �L�����d�E�
=
LayerNorm_2_bwd (0J �� �5  �LM  �L`����E�
;
DropOut_2_bwd (0J �� �5  �LM  �L`����E�
H
PROJ_GEMM_bwd (0:-���� �5  �LE @�NM  �L`������E�
I
MHA_GEMM_2_bwd1 (0B,��� �5  �LE  �LM  �N`������E�
H
MHA_GEMM_2_bwd2 (0B+��� �5  �LE  �NM  �L`�����E�
G
V_bwd	 (0:4���� �5  �LE @�NM  �L`hu  �L������E�
;
DropOut_1_bwd	 (0J �� �5  �NM  �N`����E�
9
SOFTMAX_bwd
 (0J �� �5  �NM  �N`����E�
H
MHA_GEMM_1_bwd1  (0B+��� �5  �NE  �LM  �L`�����E�
H
MHA_GEMM_1_bwd2!  (0B+��� �5  �NE  �LM  �L`�����E�
@
Q_bwd" !(0:-���� �5  �LE @�NM  �L`������E�
@
K_bwd# "(0:-���� �5  �LE @�NM  �L`������E�
X
FFN1_bwd_weight_update$ #(0B4��� ��5  �LE  �MM @�O`hu @�O��d�E����
X
FFN0_bwd_weight_update% $(0B4��� ��5  �ME  �LM @�O`hu @�O���g����
]
PROJ_GEMM_bwd_weight_update& %(0B4��� ��5  �LE  �LM @�N`hu @�N���E����
U
V_bwd_weight_update'	 &(0B4��� ��5  �LE  �LM @�N`hu @�N���E����
U
K_bwd_weight_update( '(0B4��� ��5  �LE  �LM @�N`hu @�N���E����
U
Q_bwd_weight_update) ((0B4��� ��5  �LE  �LM @�N`hu @�N���E����4-   �0=  �LE �WJPZAdd_Prev_LayerbLayerNorm_1"0=  �LE �WJPZLayerNorm_1bQ"0=  �LE �WJPZLayerNorm_1bK"0=  �LE �WJPZLayerNorm_1bV!0=  �LE ��HPZQb
MHA_GEMM_1!0=  �LE ��HPZKb
MHA_GEMM_1'0=  �NE ��JPZ
MHA_GEMM_1bSOFTMAX&0=  �NE ��JPZSOFTMAXb	DropOut_1!	0=  �LE ��HP	ZVb
MHA_GEMM_2)	0=  �NE ��JP
Z	DropOut_1b
MHA_GEMM_2)	
0=  �LE ��HPZ
MHA_GEMM_2b	PROJ_GEMM-
-   �0=  �LE �WJPZ	PROJ_GEMMb	DropOut_2)-   �0=  �LE �WJPZ	DropOut_2bAdd_1.-   �0
=  �LE �WJPZAdd_Prev_LayerbAdd_1+-   �0=  �LE �WJPZAdd_1bLayerNorm_2%0=  �LE �WJPZLayerNorm_2bFFN0#-   �0=  �ME ��IPZFFN0bGeLU0=  �ME ��IPZGeLUbFFN1(-   �0=  �LE �WJPZFFN1b	DropOut_3$0=  �LE �WJPZ	DropOut_3bAdd_2 0=  �LE �WJPZAdd_1bAdd_2+0=  �LE �WJPZLoss_bwdbDropOut_3_bwd+0=  �LE �WJPZDropOut_3_bwdbFFN1_bwd+-   �0=  �ME ��IPZFFN1_bwdbGeLU_bwd&0=  �ME ��IPZGeLU_bwdbFFN0_bwd2-   �0=  �LE �WJPZFFN0_bwdbLayerNorm_2_bwd7-   �0=  �LE �WJPZLayerNorm_2_bwdbDropOut_2_bwd00=  �LE �WJPZDropOut_2_bwdbPROJ_GEMM_bwd20=  �LE ��HPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1&0=  �LE ��HPZVbMHA_GEMM_2_bwd120=  �LE ��HPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2.0=  �NE ��JP Z	DropOut_1bMHA_GEMM_2_bwd220=  �NE ��JP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd*0=  �LE ��HP"ZMHA_GEMM_2_bwd2bV_bwd.0=  �NE ��JP#ZDropOut_1_bwdbSOFTMAX_bwd0 0=  �NE ��JP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd10!0=  �NE ��JP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2& 0
=  �LE ��HP&ZKbMHA_GEMM_1_bwd1&!0
=  �LE ��HP'ZQbMHA_GEMM_1_bwd2* "0=  �LE ��HP(ZMHA_GEMM_1_bwd1bQ_bwd*!#0=  �LE ��HP)ZMHA_GEMM_1_bwd2bK_bwd9$0=  �LE �WJP*ZDropOut_3_bwdbFFN1_bwd_weight_update0$0=  �ME ��IP+ZGeLUbFFN1_bwd_weight_update4%0	=  �ME ��IP,ZGeLU_bwdbFFN0_bwd_weight_update7%0=  �LE �WJP-ZLayerNorm_2bFFN0_bwd_weight_update>&0=  �LE �WJP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update;	&0=  �LE ��HP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE ��HP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE �WJP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE ��HP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE �WJP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE ��HP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE �WJP5ZLayerNorm_1bQ_bwd_weight_updateC�� �-   M5ff�?Z����  �C�  HB�TP�PP�
  @E  �Q-   @  �A  �?%�֋E-�d*<5��T=E��T=M�[f=U��*C"(:���� �(�X�`h�p��fff?�*
o�:�