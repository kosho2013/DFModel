
�)
8
Add_Prev_Layer(0J �� �5  �LM  �L`����r�
9
LayerNorm_1 (0J �� �5  �LM  �L`����r�
<
Q (0:-���� �5  �LE @�NM  �L`������r�
<
K (0:-���� �5  �LE @�NM  �L`������r�
<
V (0:-���� �5  �LE @�NM  �L`������r�
D

MHA_GEMM_1 (0B,��� �5  �LE  �LM  �N`������r�
5
SOFTMAX (0J �� �5  �NM  �N`����r�
7
	DropOut_1 (0J �� �5  �NM  �N`����r�
C

MHA_GEMM_2	 (0B+��� �5  �LE  �NM  �L`�����r�
K
	PROJ_GEMM
 	(0:4���� �5  �LE @�NM  �L`hu  �L������r�
7
	DropOut_2 
(0J �� �5  �LM  �L`����r�
9
Add_1	 (0R&�� �5  �LM  �L`����r��  �L
9
LayerNorm_2
 (0J �� �5  �LM  �L`����r�
D
FFN0 (0:2���� �5  �LE @�OM  �M`u   ���d����r�
1
GeLU (0J�� �5  �MM  �M`��d�r�
F
FFN1 (0:4���� �5  �ME @�OM  �L`hu  �L�����d�r�
7
	DropOut_3 (0J �� �5  �LM  �L`����r�
8
Add_2 (0R%�� �5  �LM  �L`���r��  �L
4
Loss_bwd (0J �� �5  �LM  �L`����r�
;
DropOut_3_bwd (0J �� �5  �LM  �L`����r�
C
FFN1_bwd (0:-���� �5  �LE @�OM  �M`��d����r�
5
GeLU_bwd (0J�� �5  �MM  �M`��d�r�
K
FFN0_bwd (0:5���� �5  �ME @�OM  �L`hu  �L�������r�
=
LayerNorm_2_bwd (0J �� �5  �LM  �L`����r�
;
DropOut_2_bwd (0J �� �5  �LM  �L`����r�
H
PROJ_GEMM_bwd (0:-���� �5  �LE @�NM  �L`������r�
I
MHA_GEMM_2_bwd1 (0B,��� �5  �LE  �LM  �N`������r�
H
MHA_GEMM_2_bwd2 (0B+��� �5  �LE  �NM  �L`����#�r�
G
V_bwd	 (0:4���� �5  �LE @�NM  �L`hu  �L������r�
;
DropOut_1_bwd	 (0J �� �5  �NM  �N`����r�
9
SOFTMAX_bwd
 (0J �� �5  �NM  �N`����r�
H
MHA_GEMM_1_bwd1  (0B+��� �5  �NE  �LM  �L`�����r�
H
MHA_GEMM_1_bwd2!  (0B+��� �5  �NE  �LM  �L`����!�|�
@
Q_bwd" !(0:-���� �5  �LE @�NM  �L`������r�
@
K_bwd# "(0:-���� �5  �LE @�NM  �L`������r�
X
FFN1_bwd_weight_update$ #(0B4��� ��5  �LE  �MM @�O`hu @�O��d�r����
X
FFN0_bwd_weight_update% $(0B4��� ��5  �ME  �LM @�O`hu @�O���r����
]
PROJ_GEMM_bwd_weight_update& %(0B4��� ��5  �LE  �LM @�N`hu @�N���r����
U
V_bwd_weight_update'	 &(0B4��� ��5  �LE  �LM @�N`hu @�N���r����
U
K_bwd_weight_update( '(0B4��� ��5  �LE  �LM @�N`hu @�N���r����
U
Q_bwd_weight_update) ((0B4��� ��5  �LE  �LM @�N`hu @�N���r����4-   �0=  �LE  �JPZAdd_Prev_LayerbLayerNorm_1"0=  �LE  �JPZLayerNorm_1bQ"0=  �LE  �JPZLayerNorm_1bK"0=  �LE  �JPZLayerNorm_1bV!0=  �LE  2IPZQb
MHA_GEMM_1!0=  �LE  2IPZKb
MHA_GEMM_1'0=  �NE �KPZ
MHA_GEMM_1bSOFTMAX&0=  �NE �KPZSOFTMAXb	DropOut_1!	0=  �LE  2IP	ZVb
MHA_GEMM_2)	0=  �NE �KP
Z	DropOut_1b
MHA_GEMM_2)	
0=  �LE  2IPZ
MHA_GEMM_2b	PROJ_GEMM-
-   �0=  �LE  �JPZ	PROJ_GEMMb	DropOut_2)-   �0=  �LE  �JPZ	DropOut_2bAdd_1.-   �0
=  �LE  �JPZAdd_Prev_LayerbAdd_1+-   �0=  �LE  �JPZAdd_1bLayerNorm_2%0=  �LE  �JPZLayerNorm_2bFFN0#-   �0=  �ME  2JPZFFN0bGeLU0=  �ME  2JPZGeLUbFFN1(-   �0=  �LE  �JPZFFN1b	DropOut_3$0=  �LE  �JPZ	DropOut_3bAdd_2 0=  �LE  �JPZAdd_1bAdd_2+0=  �LE  �JPZLoss_bwdbDropOut_3_bwd+0=  �LE  �JPZDropOut_3_bwdbFFN1_bwd+-   �0=  �ME  2JPZFFN1_bwdbGeLU_bwd&0=  �ME  2JPZGeLU_bwdbFFN0_bwd2-   �0=  �LE  �JPZFFN0_bwdbLayerNorm_2_bwd7-   �0=  �LE  �JPZLayerNorm_2_bwdbDropOut_2_bwd00=  �LE  �JPZDropOut_2_bwdbPROJ_GEMM_bwd20=  �LE  2IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1&0=  �LE  2IPZVbMHA_GEMM_2_bwd120=  �LE  2IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2.0=  �NE �KP Z	DropOut_1bMHA_GEMM_2_bwd220=  �NE �KP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd*0=  �LE  2IP"ZMHA_GEMM_2_bwd2bV_bwd.0=  �NE �KP#ZDropOut_1_bwdbSOFTMAX_bwd0 0=  �NE �KP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd10!0=  �NE �KP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2& 0
=  �LE  2IP&ZKbMHA_GEMM_1_bwd1&!0
=  �LE  2IP'ZQbMHA_GEMM_1_bwd2* "0=  �LE  2IP(ZMHA_GEMM_1_bwd1bQ_bwd*!#0=  �LE �AIP)ZMHA_GEMM_1_bwd2bK_bwd9$0=  �LE  �JP*ZDropOut_3_bwdbFFN1_bwd_weight_update0$0=  �ME  2JP+ZGeLUbFFN1_bwd_weight_update4%0	=  �ME  2JP,ZGeLU_bwdbFFN0_bwd_weight_update7%0=  �LE  �JP-ZLayerNorm_2bFFN0_bwd_weight_update>&0=  �LE  �JP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update;	&0=  �LE  2IP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE  2IP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE  �JP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE �AIP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE  �JP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE  2IP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE  �JP5ZLayerNorm_1bQ_bwd_weight_updateB�� -  �L5�"�?:����  HB�  HB�TP�PP�
  @E  �Q-   @  �A  �?% � G-��T=5��T=E��T=M�[f=U �;D"(:���� �(�X�`h�p��fff?�*
o�:�