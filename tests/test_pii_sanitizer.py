import unittest

from app.security import PIISanitizer


class TestPIISanitizer(unittest.TestCase):
    def test_sanitize_replaces_supported_pii_types(self) -> None:
        sanitizer = PIISanitizer()
        text = (
            "Contact Ivan Petrov by email ivan@example.com, phone +7 999 123-45-67, "
            "telegram @ivan_petrov, url https://example.com/path, uuid "
            "550e8400-e29b-41d4-a716-446655440000, ip 192.168.1.1, card "
            "4111 1111 1111 1111, inn 7707083893, snils 112-233-445 95."
        )

        sanitized = sanitizer.sanitize(text)

        self.assertIn("<NAME_1>", sanitized)
        self.assertIn("<EMAIL_1>", sanitized)
        self.assertIn("<PHONE_1>", sanitized)
        self.assertIn("<TELEGRAM_1>", sanitized)
        self.assertIn("<URL_1>", sanitized)
        self.assertIn("<UUID_1>", sanitized)
        self.assertIn("<IP_1>", sanitized)
        self.assertIn("<CARD_1>", sanitized)
        self.assertIn("<INN_1>", sanitized)
        self.assertIn("<SNILS_1>", sanitized)

    def test_restore_returns_original_text(self) -> None:
        sanitizer = PIISanitizer()
        text = "Мария Иванова: maria@example.com, +7 (999) 123-45-67, @maria_ivanova."

        sanitized = sanitizer.sanitize(text)
        restored = sanitizer.restore(sanitized)

        self.assertEqual(restored, text)

    def test_same_value_uses_same_placeholder(self) -> None:
        sanitizer = PIISanitizer()

        sanitized = sanitizer.sanitize("a@test.com and a@test.com")

        self.assertEqual(sanitized, "<EMAIL_1> and <EMAIL_1>")
        self.assertEqual(sanitizer.mapping, {"<EMAIL_1>": "a@test.com"})

    def test_restore_unknown_placeholder_keeps_text_unchanged(self) -> None:
        sanitizer = PIISanitizer()

        self.assertEqual(sanitizer.restore("Keep <EMAIL_999>"), "Keep <EMAIL_999>")


if __name__ == "__main__":
    unittest.main()
