from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.stacklayout import StackLayout

from app.components.annotating.token import Token

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
    sentence = ObjectProperty(None, allownone=True)

    def on_sentence(self, instance, value):
        self.update_tokens()

    def update_tokens(self):
        self.clear_widgets()
        if self.sentence:
            for token in self.sentence.tokens:
                if token.words:
                    self.add_widget(Token(annotation=token))
