
�
8
Add_Prev_Layer(0J �� �5  zNM  zN`����0�
9
LayerNorm_1 (0J �� �5  zNM  zN`����0�
=
Q (0:.���� �5  zNE $�QM  zN`�������0�
=
K (0:.���� �5  zNE $�QM  zN`�������0�
=
V (0:.���� �5  zNE $�QM  zN`�������0�
D

MHA_GEMM_1 (0B,��� �5  zNE  zNM  �O`���
���0�
5
SOFTMAX (0J �� �5  �OM  �O`���
�0�
7
	DropOut_1 (0J �� �5  �OM  �O`���
�0�
D

MHA_GEMM_2	 (0B,��� �5  zNE  �OM  zN`������0�
L
	PROJ_GEMM
 	(0:5���� �5  zNE $�QM  zN`hu  zN�������0�
7
	DropOut_2 
(0J �� �5  zNM  zN`����0�
9
Add_1	 (0R&�� �5  zNM  zN`����4��  zN
9
LayerNorm_2
 (0J �� �5  zNM  zN`����0�
@
FFN0 (0:.��>�� �5  zNE $�RM  zO`�������0�
2
GeLU (0J ��> �5  zOM  zO`����0�
G
FFN1 (0:5����> �5  zOE $�RM  zN`hu  zN�������/�
7
	DropOut_3 (0J �� �5  zNM  zN`����0�
:
Add_2 (0R'�� �5  zNM  zN`�������  zN7-   �0=  zNE ��KPZAdd_Prev_LayerbLayerNorm_1�%0=  zNE���KPZLayerNorm_1bQ�%0=  zNE���KPZLayerNorm_1bK�%0=  zNE���KPZLayerNorm_1bV�$0=  zNE��;JPZQb
MHA_GEMM_1�$0=  zNE��;JPZKb
MHA_GEMM_1�*0=  �OE  pKPZ
MHA_GEMM_1bSOFTMAX�)0=  �OE  pKPZSOFTMAXb	DropOut_1�$	0=  zNE �;JP	ZVb
MHA_GEMM_2�,	0=  �OE  pKP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  zNE �;JPZ
MHA_GEMM_2b	PROJ_GEMM�0
-   �0=  zNE`��KPZ	PROJ_GEMMb	DropOut_2�,-   �0=  zNE ��KPZ	DropOut_2bAdd_1�1-   �0
=  zNE ��KPZAdd_Prev_LayerbAdd_1�.-   �0=  zNE  �KPZAdd_1bLayerNorm_2�--   �0=  zNE ��KPZLayerNorm_2bFFN0�&-   �0=  zOE �;KPZFFN0bGeLU�&-   �0=  zOE �;KPZGeLUbFFN1�+-   �0=  zNE ��KPZFFN1b	DropOut_3�,-   �0=  zNE ��KPZ	DropOut_3bAdd_2�(-   �0=  zNE  �KPZAdd_1bAdd_2�-�  -  PNU���?"��  zD�TP�  �D2   @  �A  �?%��F-��T=5��T==��T=E��T=M(a&>UeZ�C"":���� �(�X��`xp��*o�:�