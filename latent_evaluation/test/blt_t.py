from pipeline.encoder import BLTEncoder

enc = BLTEncoder()
enc.load_model()
print(enc.encode_text("TEXT")['patch_embeddings'])
print(enc.encode_text("TEXT")['patch_embeddings'].shape)