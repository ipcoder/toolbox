###

```plantuml


dict <|- Labels

class Labels {
    attr access

    to_keys()
    to_index()
    tuple()
}


Labels <|- Labeled

class Labeled {
    defined Labels

    from_table(cls, DataTable):

    flat(): str
}

metaclass LabeledType {

}

DataC -u-|> Labeled
DataC <-l- LabeledType

```