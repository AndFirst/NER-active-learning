from kivy.core.window import Window


def is_key_pressed(keyname: str):
    return keyname in Window.modifiers
