from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from app.components.add_labels.label_row import LabelRow

kv_string = """
<AddLabelForm>:
    orientation: 'vertical'
    padding: 20
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Line:
            rectangle: (self.x, self.y, self.width, self.height)
            width: 1
    ScrollView:
        id: scroll_view
        scroll_type: ['bars']
        size_hint: 1, 1
        bar_width: 20
        scroll_y: 1.0
        # canvas.before:
        #     Color:
        #         rgba: 0, 1, 0, 1
        #     Line:
        #         rectangle: (self.x, self.y, self.width, self.height)
        #         width: 1
        GridLayout:
            id: layout
            spacing: 20
            cols: 1
            size_hint_y: None
            height: self.minimum_height
            # canvas.before:
            #     Color:
            #         rgba: 1, 0, 0, 1
            #     Line:
            #         rectangle: (self.x, self.y, self.width, self.height)
            #         width: 1
            LabelRow:


"""

Builder.load_string(kv_string)


class AddLabelForm(BoxLayout):
    @property
    def label_rows(self):
        label_rows = [
            child
            for child in self.ids.layout.children
            if isinstance(child, LabelRow)
        ]
        return label_rows
