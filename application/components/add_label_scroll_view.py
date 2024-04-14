from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from components.label_row import LabelRow


class AddLabelScrollView(ScrollView):
    def __init__(self, **kwargs):
        super(AddLabelScrollView, self).__init__(**kwargs)
        self.bar_width = 20
        self.scroll_type = ['bars']

        self.labels_layout = BoxLayout(
            orientation='vertical', spacing=5, size_hint_y=None)
        self.labels_layout.bind(
            minimum_height=self.labels_layout.setter('height'))

        self.add_widget(self.labels_layout)

    def add_label_row(self, on_add_row, on_delete_row, text=''):
        new_row = LabelRow(on_add_row=on_add_row,
                           on_delete_row=on_delete_row, text=text)
        self.labels_layout.add_widget(new_row)
        return new_row

    def delete_label_row(self, row):
        if len(self.labels_layout.children) > 1:
            self.labels_layout.remove_widget(row)
        else:
            print("Cannot remove all labels")
