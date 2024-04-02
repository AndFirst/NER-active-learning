from kivy.uix.textinput import TextInput
from kivy.lang import Builder

kv_string = """
<LabelInput>:
    multiline: False
    on_text_validate: root.on_text_validate()
    background_color: (1, 1, 1, 1) if len(self.text.split()) <= 1 else (1, 0, 0, 1)
"""

Builder.load_string(kv_string)


class LabelInput(TextInput):
    def __init__(self, on_add_row, **kwargs):
        super(LabelInput, self).__init__(**kwargs)
        self.on_add_row = on_add_row
        self.multiline = False
        self.bind(text=self.check_input)

    def insert_text(self, substring, from_undo=False):
        if substring.isalnum() or substring in "-_":
            # Allow alphanumeric characters and specific special characters
            return super(LabelInput, self).insert_text(substring, from_undo=from_undo)
        else:
            # Reject other characters
            return super(LabelInput, self).insert_text('', from_undo=from_undo)

    def on_text_validate(self, *args):
        if self.text.strip():
            self.on_add_row()

    def set_focus(self):
        self.focus = True

    def check_input(self, instance, value):
        if len(value.split()) > 1:
            self.background_color = (1, 0, 0, 1)  # Red background for error
        else:
            # White background for no error
            self.background_color = (1, 1, 1, 1)
