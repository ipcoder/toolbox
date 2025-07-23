# Data Casting
## The Problem
The *Data Casting* comes to solve problems of virtually unlimited multiplicity of
different ways used to organize datasets in files.

Additional challenges are related to existence of *implicit information* associated with datasets:

- special way of encoding additional meta information in files and folders names
- particular meaning hidden in conventions of their placement
- assumptions about data formats, encodings, order, units
- and many more

The common approach of creating *ad hock* "readers" for every kind of dataset
is characterized by multiple issues:

1. The code of parsing such files tree is cumbersome and prone to errors
2. It is usually not too robust toward changes in the content of the folders
3. Resulting structure of read the data is decided ad hoc as well and often suitable
   only to the single usage scenario it was initially intended for.
4. When the usage or source dataset changes, its often seems preferable to create
   hacks around existing code, or produce its slightly different copies, -
   all leading to multiplication of bugs and maintenance effort.
5. All the implicit information is hard coded or remains implicit in the resulting data
6. Parsing the content of large datasets (> 10^5) files takes significant time spent on
   every time access, unless (another non-portable ad hoc) caching is implemented.
7. All the knowledge collected to create such "readers" is **implicitly**
   buried in the code and is hardly transferable to new developers.

## The Data Casting Approach

###  Main Features
1. ALL the information describing datasets is **explicit**
2. This description is *declarative* and separated from the *generic parsing code*
3. Datasets are cast into same *universal data structure*
4. A *generic caching* mechanism is built into the Data Caster

### Primary Concepts
This approach is developed based on the following basic concepts:

- *datasource*
- *layout*
- *scheme*
- *labeled data*
- *dataset*

A `datasource` is a folder containing files of a dataset organized in accordance with of certain
set of principles called `layout` of this dataset.
There could be multiple `datasources` following same `layout`.

> *Note*. We use `layout` as an identifier of a particular data organization,
rather than its formal description.

Instead, we introduce `scheme` as *formal description* of a different kind.

It defines how datasources of specific `layout` should be interpreted and cast
into the *universal form of labeled data*.

Normally a ``scheme`` is designed for a particular `layout`.
It's not uncommon to have *multiple* schemes associated with the same `layout`.

```plantuml
left to right direction

folder "datasource X: B" as DX
folder "datasource Y: B" as DY
folder "datasource Z: B" as DZ
hexagon "layout B" as LB

file "Scheme 1: B" as S_B1
file "Scheme 2: B" as S_B2

DX --> LB
DY --> LB
DZ --> LB

LB --> S_B1
LB --> S_B2
```
#### Labeled Data

- Every single data item in the dataset is represented by collection of ``labels`` (key-value pairs).
- Keys are called ``categories`` and can be any strings
- Values could be of any type
- Each application domain may use its own vocabulary for ``categories``.
- Some categories, like `path`, `data`, are common and contain correspondingly
  the path to the data file and its content
- A particular data item is generally labeled by a subset of the `categories`

#### Dataset
Both the result and the recipie of producing labeled data from a *single* `datasource`
is called a `dataset`.

```plantuml
left to right direction
folder datasource
node "Data Casting" as Casting
database dataset
file dataset as dc

datasource --> Casting
dc -l-> Casting
Casting --> dataset
```

#### Filtering
Filtering allow to select only a part of data items produced from a `datasource`.
It's the final stage of data casting pipeline,
so its query is defined in terms of already produced labels.

>  Filtering is particularly useful when reduction in dataset items is significant (N_f << N).
>
> While the same query could be applied later to the `data collection` built from the whole `datasource`,
> the performance hit would be significant: O(N_f) ⇒ 2 x O(N).
>
> Additional benefits of filtered `datasets` over filtered `data collections` is apparent when
looking at them as [resources](data_resources.md).
Datasets are convenient *stable* units of data which can be combined into different data collections.

## The Casting

The purpose of a ``scheme`` is to define how the content of a ``datasource`` should be cast into the
labeled data items.

Resulting labels must be:
 1. **Domain Compatible** - use categories, units, formats adopted by specific application domain
 2. **Complete** - contain all the information about semantic meaning of the data

Compatibility is ensured by designing scheme which uses categories standard for the domain and
maps them into correct data attributes.

Completeness is considered from the usage point of view,
so that a simplified schemes can be created for different purposes.

The transformation includes two main stages: **labeling** and **filtering**.

```plantuml
    skinparam defaultFontSize 16

    card DataCast {
        storage Labeling #line.dotted {
            left to right direction
            control parsing
            stack mapping
            hexagon labeling
            interface " " as xx
        }
        queue Filter
    }

    folder datasource
    collections labels
    file scheme

    scheme -r-> Labeling

    parsing --> mapping
    datasource --> parsing
    mapping -l-> xx
    labeling --> xx
    xx --> Filter
    Filter --> labels
```

#### The Pattern
At the heart of every scheme is a path matching pattern, generally represented by a regex pattern
designed to extract **named groups** following the logic of specific ``layout``.

Names of those groups are selected to represent the correct `categories`.

Paths matching produces initial labels for every file, automatically leaving other files out.

> *Note*. Pattern is matched with a *part of the file's path* relative to the `datasource` root folder.

The values of the matched groups (parts of the path string) may require additional ``mapping``
to comply with domain conventions.


#### Mapping Labels' Values
Let's examine a segment of a scheme parsing layouts with paths like `vid_6/Cam_1/rgb_86.bmp`.
(schemes are usually represented by YAML files)
```yaml
pattern: 'vid_(?P<scene1>\d+)/(?P<alg>\w+)_(?P<view>\d+)/(?P<kind>\w+)_(?P<scene2>\d+)\.(?P<ext>\w+)'

mappings:
    view: {"0": R, "1": L}
    kind: {rgb: image}
    alg: str.lower
```

Named groups in the ``pattern`` define categories: `scene1, alg, view, kind, scene2, ext`.
The `mapping` section makes sure that:
 1. category `view` instead of captured values `"0", "1"` is assigned by
`"R", "L"` as required by the _stereo depth domain conventions_.
 2. `kind` labels are replaced: `{"kind": "rgb"} ⇒ {"kind": "image"}`)
 3. values for `alg` are mapped using a different mechanism of applying a python function
    `str.lower`. Any python function is supported.

### Adding Labels Explicitly

When paths do not contain some relevant implicit information, scheme allows to provide
it explicitly using additional labels.

Any YAML key-node which is not a *reserved keyword*,
is considered a label and is **added to ALL the data items** of the produced dataset.

In example below it would be `{"synthetic": True}`.\
Alternatively all the nodes under `labels` node are considered new labels.
That makes `{"realism": "low"}` another global label.

Sometimes it could be a preferable practice to keep on the top level only the reserved keywords.

#### Conditional Labels
Some labels must be assigned only to a specif data items. \
That can be accomplished using *conditional nodes* following the template: ``if <expression>``.

Here `<expression>` is any python expression evaluated in the
namespace filled by all the defined categories as variables.

Thus, in this example new conditional labels are added for items with `kind == 'image'`:
`{"alg": "cam", "range" :"0,255"}`.
```yaml
name: FT3D              # reserved field

search:                 # reserved field
    max_depth: 10

pattern: "(?P<kind>(frames_cleanpass)|(disparity)|(occlusions))/{subset}\
          /{scene_1}/{scene_2}/{view}/{scene_3}\
          \\.(?P<ext>(?(3)pfm|png))"

synthetic: true         # <-- new label

labels:                 # reserved field
    realism: low        # <-- new label

mappings:               # reserved field
    subset: str.lower
    kind: { frames_cleanpass: image, disparity: disp, occlusions: occl }
    view: { left: L, right: R }

if kind == 'image':     # reserved field
    alg: cam            # <-- new conditional label
    range: "0,255"      # <-- new conditional label


if kind in ('disp', 'occl'):  # reserved field
    alg: GT                   # <-- new conditional label

_samples:               # reserved field
    - disparity/TRAIN/A/0749/right/0015.pfm
    - frames_cleanpass/TEST/C/0145/right/0013.png

```
> *Note 1*. Example above also illustrates an alternative way to define named group in regular expression
> supported by this syntax: `{category}` is a shortcut for `(?P<category>\w+)`.

> *Note 2*. **Automatic labels merging** is applied to labels with enumerated names:\
> `<common>_1: <value1>, <common>_2: <value2>, ...`\
> Resulting label value is merged using the `_` separator: `<common>: <value1>_<value2>_...`

> *Note 3*. Node `_samples` contain samples of typical paths in this layout for automatic pattern validation.

> *Note 4*. Every data item is automatically labeled with: `path: "<fill_file_path>"`

-----
Read also about using the elements discussed here as [resources](data_resources.md).