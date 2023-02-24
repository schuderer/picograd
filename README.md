# picograd
Attempt to reproduce Andrej Karpathy's migrograd auto differentiation engine in C++.

I am an absolute newbie at C++, so please do not use this code for anything important. :)

`Value` is the user-facing class (meant for being used as a stack object by the user). There's a
hidden `Node` graph behind it (every `Value` has an underlying `Node`), and `Nodes` can live
on if referenced from other nodes in the graph. `std::shared_ptr` seems to work nicely here
and there don't seem to be any memory management issues
(You can tell that I still have a novice's confidence. Famous last words.).

This being a template class, I was torn about file structure. Currently, the last line of
`picograd.h` imports `picograd.cpp`.
In Qt Creator, this causes cyclic import warnings. If these annoy you, pasting the contents
of `picograd.cpp` at the spot of the import and removing its own import of `picograd.h` should fix it.

The main.cpp just contains some tests and debugging stuff. You can scrap all of this
(please don't don't hate me for my `delete` override causing a crash at the end ;) ).

