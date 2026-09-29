from pipeline.encoder import BolmoEncoder

enc = BolmoEncoder()
enc.load_model()

print("Encoded")
print(enc.encode_text("TEXT")['patch_embeddings'])
print(enc.encode_text("TEXT")['patch_embeddings'].shape)
print("\n")

print("Tokenized")
input_ids = enc.tokenizer(["TEXT"], return_tensors="pt", add_special_tokens=False)["input_ids"]
print(input_ids.shape)
input_ids = input_ids[: , 1:]
print(input_ids.shape)
input("Press Enter to continue...")




enc.delete_model()

input("Press Enter to continue...")