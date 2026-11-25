# assets/

Files the Lap replay player embeds. replay.py reads them, so nothing here is
fetched from the internet when the page runs.

## fonts/

- `Archivo-latin-var.woff2`: Archivo, a variable font (weight and width) by
  Omnibus-Type, free under the SIL Open Font License 1.1
  (https://openfontlicense.org; the full text is in `fonts/OFL.txt`). Used at 75 % width and extra bold for the
  driver numbers on the car markers, and for the player's text. Latin subset.

### Using a licensed team font

1. Put the `.woff2` file in `fonts/`.
2. In `replay.py`, change `NUMBER_FONT`: the file path, the font's family
   name, and `settings` (`"normal"` for a font that isn't variable).
3. Only commit a font you hold a licence for. Team fonts and logos are
   their teams' property and aren't included here.

## images/ (not created yet)

The slot for licensed team images or logos, if you add them later. Nothing
uses this folder yet.
