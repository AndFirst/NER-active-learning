# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.button import Button
# from components.add_label_scroll_view import AddLabelScrollView
# from kivy.uix.popup import Popup
# from kivy.uix.label import Label
# from collections import Counter
# from kivy.clock import Clock


# class AddLabelForm(BoxLayout):
#     def __init__(self, **kwargs):
#         super(AddLabelForm, self).__init__(**kwargs)
#         self.orientation = 'vertical'
#         self.padding = 50
#         # Scroll View dla etykiet
#         self.scroll_view = AddLabelScrollView()
#         self.add_widget(self.scroll_view)

#         # Kontener dla przycisków
#         button_container = BoxLayout(
#             orientation='horizontal', size_hint_y=None, height=40)

#         # Przycisk zapisywania
#         self.save_button = Button(text="Zapisz", size_hint_x=None, width=100)
#         self.save_button.bind(on_press=self.save)
#         button_container.add_widget(self.save_button)

#         self.add_widget(button_container)

#         # Dodanie wiersza początkowego
#         self.add_table_row()

#     def add_table_row(self):
#         # Sprawdź, czy istnieje pusty wiersz
#         empty_row = None
#         for child in self.scroll_view.labels_layout.children:
#             if child.text_input.text.strip() == '':
#                 empty_row = child
#                 break

#         # Jeśli istnieje pusty wiersz, przenieś na niego focus
#         if empty_row:
#             empty_row.text_input.focus = True
#         else:
#             new_row = self.scroll_view.add_label_row(
#                 on_add_row=self.add_table_row,
#                 on_delete_row=self.delete_label_row,
#                 text=''
#             )
#             new_row.text_input.set_focus()

#     def validate_labels(self, labels):
#         errors = []

#         counter = Counter(labels)
#         duplicated_labels = [label for label,
#                              count in counter.items() if count > 1]
#         if "" in labels:
#             errors.append("Empty label. Please remove it from list.")
#         elif duplicated_labels:
#             errors.append(
#                 f"Please provide unique labels. Duplicated labels: \n{duplicated_labels}")
#         return errors

#     def show_validation_errors(self, errors):
#         if errors:
#             print(errors)
#             error_message = "\n".join(errors)
#             popup = Popup(title='Błędy walidacji', content=Label(
#                 text=error_message), size_hint=(None, None), size=(600, 400))

#             # Ustawienie zamykania popupu po 5 sekundach
#             Clock.schedule_once(lambda dt: popup.dismiss(), 5)
#             popup.open()

#     def delete_label_row(self, widget):
#         self.scroll_view.delete_label_row(widget)

#     def save(self, instance):
#         labels = [child.text_input.text.strip().lower()
#                   for child in self.scroll_view.labels_layout.children]
#         errors = self.validate_labels(labels)
#         if errors:
#             self.show_validation_errors(errors)
#         else:
#             print(labels)
