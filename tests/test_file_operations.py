from io import StringIO
from app.file_operations import remove_sentence_from_csv


def test_remove_sentence():
    input_data = (
        """1,John,Doe\n2,Jane,Smith\n3,Bob,Johnson\n4,Alice,Williams\n"""
    )
    input_file = StringIO(input_data)

    expected_output = """1,John,Doe\n2,Jane,Smith\n4,Alice,Williams\n"""

    output_file = StringIO()

    remove_sentence_from_csv(2, input_file, output_file)

    output_file.seek(0)

    # Normalize newline characters for comparison
    output_data = output_file.getvalue()

    assert output_data == expected_output
