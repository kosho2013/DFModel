
�
9
Add_Prev_Layer(0J!�� �5  zNM  zN`������
:
LayerNorm_1 (0J!�� �5  zNM  zN`������
>
Q (0:/���� �5  zNE $�QM  zN`���������
>
K (0:/���� �5  zNE $�QM  zN`���������
>
V (0:/���� �5  zNE $�QM  zN`���������
E

MHA_GEMM_1 (0B-��� �5  zNE  zNM  �O`���
�����
6
SOFTMAX (0J!�� �5  �OM  �O`�����
8
	DropOut_1 (0J!�� �5  �OM  �O`���
���
E

MHA_GEMM_2	 (0B-��� �5  zNE  �OM  zN`��������
M
	PROJ_GEMM
 	(0:6���� �5  zNE $�QM  zN`hu  zN���������
8
	DropOut_2 
(0J!�� �5  zNM  zN`������
:
Add_1	 (0R'�� �5  zNM  zN`�������  zN
:
LayerNorm_2
 (0J!�� �5  zNM  zN`������
A
FFN0 (0:/��>�� �5  zNE $�RM  zO`���������
3
GeLU (0J!��> �5  zOM  zO`������
H
FFN1 (0:6����> �5  zOE $�RM  zN`hu  zN���������
8
	DropOut_3 (0J!�� �5  zNM  zN`������
:
Add_2 (0R'�� �5  zNM  zN`�������  zN7-   �0=  zNE ��LPZAdd_Prev_LayerbLayerNorm_1�%0=  zNE ��LPZLayerNorm_1bQ�%0=  zNE ��LPZLayerNorm_1bK�%0=  zNE ��LPZLayerNorm_1bV�$0=  zNE �;KPZQb
MHA_GEMM_1�$0=  zNE �6KPZKb
MHA_GEMM_1�*0=  �OE  pLPZ
MHA_GEMM_1bSOFTMAX�)0=  �OE<Y�LPZSOFTMAXb	DropOut_1�$	0=  zNEv�6KP	ZVb
MHA_GEMM_2�,	0=  �OE �vLP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  zNE �6KPZ
MHA_GEMM_2b	PROJ_GEMM�0
-   �0=  zNE/��LPZ	PROJ_GEMMb	DropOut_2�,-   �0=  zNE ��LPZ	DropOut_2bAdd_1�1-   �0
=  zNE ��LPZAdd_Prev_LayerbAdd_1�.-   �0=  zNE ��LPZAdd_1bLayerNorm_2�--   �0=  zNE ��LPZLayerNorm_2bFFN0�&-   �0=  zOE �6LPZFFN0bGeLU�&-   �0=  zOE �6LPZGeLUbFFN1�+-   �0=  zNE ��LPZFFN1b	DropOut_3�,-   �0=  zNE ��LPZ	DropOut_3bAdd_2�(-   �0=  zNE ��LPZAdd_1bAdd_2�-�  -  PNU���?"��  zD�TP� P�G2   @  �A  �?%��F-��T=5��T==��T=E��T=M(a&>UeZ�C"":���� �(�X��`xp��*o�:�