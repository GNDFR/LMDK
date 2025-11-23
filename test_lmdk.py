# test_lmdk.py
import os
import pytest
from lmdk import cli
from typer.testing import CliRunner

# Helper function to create a temporary file with content
@pytest.fixture
def temp_file(tmpdir):
    def _temp_file(content):
        file = tmpdir.join("test.txt")
        file.write(content)
        return str(file)
    return _temp_file

# Test the `prep` command
def test_prep_command(temp_file, tmpdir):
    # Create a dummy file with some content
    filepath = temp_file("This is a test sentence.\nThis is another test sentence.\nThis is a bad sentence with a badword1.")

    # Create a dummy toxic keywords file
    toxic_filepath = tmpdir.join("toxic.txt")
    toxic_filepath.write("badword1")

    # Run the `prep` command
    runner = CliRunner()
    output_filepath = "cleaned_output.txt"
    result = runner.invoke(cli.app, ["prep", os.path.abspath(filepath), "--toxic-keywords-file", os.path.abspath(toxic_filepath), "--output-file", output_filepath])

    # Check that the command ran successfully
    assert result.exit_code == 0

    # Check that the output file contains the correct number of lines
    with open(output_filepath, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
