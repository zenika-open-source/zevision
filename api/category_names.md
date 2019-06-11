# Definition

In the context of Face Recognition, a _category_ is a person name or, eventually, any ID string that allows to identify a person in a non-ambiguous way.

To avoid ambiguity between first and last names, we recommend using `lastname_firstname_middlename` _(`nom_prenom`)_ pattern, and better yet, using ASCII-only variants without whitespaces.
E.g. a person name like Keizer Söze would be transformed into a category name - `soze_keizer`.
