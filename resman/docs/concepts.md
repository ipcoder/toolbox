# Main Concepts of Resource Management

## Resource Definition
_Resource_ is an information unit of hierarchical *attributes* managed by the Resource Management framework.
 - Every resource belongs to specific *Resource Type* defined by its *model*.
 - Resources are *named*, *locatable* and *referenceable*.

## Resource Representation
Every resource is _represented_ by an instance of specific **pydantic** model class which defines its type,
together with its attributes structure and possible validity conditions.

Resources are _retained_ as YAML files in configurable system of locations to make them *locatable*.

### Special attributes
```yaml
name: [mandatory], unique within its type

# Internal, can be optionally generated automatically for integrity checks
_model:
    class:  name of the model
    ver:    version of the model

# Automatically generated when loaded from file
_file:
    folder: absolute path to the location of this file
    name:   file name name without extension
    suffix: suffix of this file
```
Other attributes may contain arbitrary data items (as describable by pydantic model),
leaving its interpretation to a *Recipient* of the resource.

## Resource Management
That includes the following main capabilities:
1. Maintain dynamic in-memory registry of discovered resources
2. Distinguish resources by their **types**
3. Ensure uniqueness of resources names within their type
4. Locating resources by their names and types

## Resource Location
Generally resource files may be located in any place,
but framework provides certain facilities to simplify their organization and search.

Every resource type (through its model class) is associated with a list of folders.
For portability reason it's a good practice to define them relative to
some *resource root* folder defined by system environment variables.
