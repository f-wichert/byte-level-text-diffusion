import re

from hat_splitter import HATSplitter as RustHATSplitter


class HATSplitter:
    def __init__(self, special_token_dict: dict | None = None, max_word_size: int = 128):
        self.hat_splitter = RustHATSplitter()
        self.max_word_size = max_word_size
        self.special_token_dict = special_token_dict
        self.special_token_replace: dict[int, list[int]] = {token: list(text.encode("utf-8")) for text, token in self.special_token_dict.items()}
        self.special_token_pattern = re.compile(rf"({'|'.join(map(re.escape, special_token_dict.keys()))})") if special_token_dict else re.compile(r"(?!)")

    def encode(self, text: str) -> list[list[int]]:
        chunks = []
        for str_chunk in self.special_token_pattern.split(text):
            if str_chunk:
                if str_chunk in self.special_token_dict:
                    chunks.append([self.special_token_dict[str_chunk]])
                else:
                    chunks.extend(list(chunk) for chunk in self.hat_splitter.split_with_limit(str_chunk, self.max_word_size))
        return chunks

    def decode(self, token_ids: list[int], errors: str = "replace", skip_special_tokens: bool = False) -> str:
        assert isinstance(token_ids, list), "token_ids must be a list"
        assert all(isinstance(token_id, int) for token_id in token_ids), "token_ids must be a list of integers"

        new_token_ids: list[int]
        if skip_special_tokens:
            new_token_ids = [token_id for token_id in token_ids if token_id not in self.special_token_replace]
        else:
            new_token_ids = []
            for token in token_ids:
                if token in self.special_token_replace:
                    new_token_ids.extend(self.special_token_replace[token])
                else:
                    new_token_ids.append(token)

        return bytes(new_token_ids).decode("utf-8", errors=errors)
