
�
5
Add_Prev_Layer(0J�@ 5  �GM  �G`��@��
4
LayerNorm_1(0J�@ 5  �GM  �G`��@��
<
Q(0:/@��@ 5  �GE   MM  �G`����@���   K
<
K(0:/@��@ 5  �GE   MM  �G`����@���   K
<
V(0:/@��@ 5  �GE   MM  �G`����@���   K
F
K_cache(0J3@� � 5  �GM  �L`���� ��  �J��  �J��  �J
F
V_cache(0J3@� � 5  �GM  �L`���� ��  �J��  �J��  �J
?

MHA_GEMM_1(0B)@� � 5  �GE  �LM   J`����� �
0
SOFTMAX	(0J@ � 5   JM   J`��� �
2
	DropOut_1
(0J@ � 5   JM   J`��� �
?

MHA_GEMM_2(0B)@��  5  �LE   JM  �G`���� ��
K
	PROJ_GEMM(0:6�@�@ 5  �GE   MM  �G`hu  �G��@�����   K
2
	DropOut_2	(0J�@ 5  �GM  �G`��@��
4
Add_1
(0R#�@ 5  �GM  �G`��@���  �G
4
LayerNorm_2(0J�@ 5  �GM  �G`��@��
@
FFN0(0:0���@ 5  �GE  �MM  �H`����@���  �K
.
GeLU(0J�� 5  �HM  �H`����
G
FFN1(0:7�@�� 5  �HE  �MM  �G`hu  �G��@�����  �K
2
	DropOut_3(0J�@ 5  �GM  �G`��@��
4
Add_2(0R#�@ 5  �GM  �G`��@���  �G7-   �0=  �GE  �GPZAdd_Prev_LayerbLayerNorm_1�%0=  �GE  �GPZLayerNorm_1bQ�%0=  �GE  �GPZLayerNorm_1bK�%0=  �GE  �GPZLayerNorm_1bV�!0=  �GE  �EPZKbK_cache�!0=  �GE  �EPZVbV_cache�$0=  �GE  �EPZQb
MHA_GEMM_1�)0=  �LPZK_cacheb
MHA_GEMM_1���*	0=   JE   HP	Z
MHA_GEMM_1bSOFTMAX�)	
0=   JE   HP
ZSOFTMAXb	DropOut_1�,
0=   JE   HPZ	DropOut_1b
MHA_GEMM_2�)0=  �LPZV_cacheb
MHA_GEMM_2���,0=  �GE  �EPZ
MHA_GEMM_2b	PROJ_GEMM�0-   �0=  �GE  �GPZ	PROJ_GEMMb	DropOut_2�,-   �0=  �GE  �GPZ	DropOut_2bAdd_1�.-   �0=  �GE  �GPZAdd_1bLayerNorm_2�1-   �0=  �GE  �GPZAdd_Prev_LayerbAdd_1�--   �0=  �GE  �GPZLayerNorm_2bFFN0�&-   �0=  �HE  �FPZFFN0bGeLU�&-   �0=  �HE  �FPZGeLUbFFN1�+-   �0=  �GE  �GPZFFN1b	DropOut_3�,-   �0=  �GE  �GPZ	DropOut_3bAdd_2�(-   �0=  �GE  �GPZAdd_1bAdd_2�8�  -  NU���?�E  C"��  HB�TP�
���D  �Q(   @  �A  �?% � G-�d*<E��T=M(a&>U �;D"):�@�@ (PX��`xpx��fff?��*o�:�