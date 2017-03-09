#!/bin/sh

printf "sage: preparsing carleman_core.sage ... "
sage --preparse carleman_core.sage
printf "done\n"
mv carleman_core.sage.py carleman_core.py

printf "sage: preparsing carleman_utils.sage ... "
sage --preparse carleman_utils.sage
printf "done\n"
mv carleman_utils.sage.py carleman_utils.py
