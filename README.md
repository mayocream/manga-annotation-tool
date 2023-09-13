# Manga Annotation Tool (MAT)

The tool is created to annotate 吹き出し・セリフ・擬声語 on manga images automatically and under the supervision of human, which is useful for the creation of high quality dataset for research.

## Annotate with textless version

無字差分 (textless version) is usally provied with the original CG・イラスト. This is helpful and make it easier to create masks for speech bubbles and texts with image processing libraries.

The Python script `annotate_with_textless.py` can be used to annotate the text version with the textless version.
