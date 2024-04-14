from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.stacklayout import StackLayout

from components.token import Token

kv_string = """
<AnnotationContainer>:
    padding: 10
    spacing: 10
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1
    """
Builder.load_string(kv_string)


class AnnotationContainer(StackLayout):
    sentence = ObjectProperty(None)
    annotation_callback = ObjectProperty(None)

    def on_sentence(self, instance, value):
        self.update_tokens()

    def update_tokens(self):
        self.clear_widgets()
        for token in self.sentence.tokens:
            self.add_widget(
                Token(
                    annotation=token,
                    update_form_state=self.annotation_callback,
                )
            )
