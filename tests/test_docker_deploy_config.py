from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DockerDeployConfigTests(unittest.TestCase):
    def test_dockerfile_runs_gradio_on_container_port(self):
        dockerfile = (ROOT / "Dockerfile").read_text()

        self.assertIn("EXPOSE 7860", dockerfile)
        self.assertIn("GRADIO_SERVER_NAME=0.0.0.0", dockerfile)
        self.assertIn('"python"', dockerfile)
        self.assertIn('"-m"', dockerfile)
        self.assertIn('"app.ui"', dockerfile)

    def test_compose_uses_env_file_and_persistent_data_volume(self):
        compose = (ROOT / "docker-compose.yml").read_text()

        self.assertIn("env_file:", compose)
        self.assertIn(".env", compose)
        self.assertIn("7860:7860", compose)
        self.assertIn("./data:/app/data", compose)

    def test_dockerignore_excludes_secrets_and_local_artifacts(self):
        dockerignore = (ROOT / ".dockerignore").read_text()

        self.assertIn(".env", dockerignore)
        self.assertIn(".venv", dockerignore)
        self.assertIn("data/deals.sqlite", dockerignore)
        self.assertIn("data/products_vectorstore", dockerignore)


if __name__ == "__main__":
    unittest.main()
