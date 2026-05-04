import re

from pygraphviz import AGraph


class IGraph(AGraph):
    """ See ./tests/test_dotsyle.py for examples """

    def __init__(self, thing=None, filename=None, data=None,
                 string=None, handle=None, name='', strict=True,
                 directed=False, styles=None, **attr, ):
        """
        Add styles kw to AGRaph support DotStyle aliases in the dot format.
        """
        self.styles = {} if styles is None else styles

        if string:
            for s in self.styles:
                string = re.sub(fr'(\[[^\]]*?){s}\b', fr'\1{styles[s].to_str()}', string)
        #         print(string)
        super().__init__(thing=thing, filename=filename, data=data, string=string,
                         handle=handle, name=name, strict=strict, directed=directed, **attr)

    def copy(self):
        from copy import deepcopy
        res = super().copy()
        res.styles = deepcopy(self.styles)
        return res

    def _repr_svg_(self):
        return str(self.draw(prog='dot', format='svg'))

    def select_edges(self, chain: str):
        for p in '-,>':
            chain = chain.replace(p, ' ')
        chain = chain.split()
        return [self.get_edge(*e) for e in zip(chain[:-1], chain[1:])]

    def style_edges(self, edges, attrs):
        if isinstance(edges, str):
            edges = self.select_edges(edges)
        if isinstance(attrs, str):
            attrs = self.styles[attrs]
        for e in edges:
            e.attr.update(attrs)

    def delete_node_edges(self, node, in_edges=True, out_edges=True):
        """ Delete the node with its edges."""
        if in_edges: self.delete_edges_from(self.in_edges(node))
        if out_edges: self.delete_edges_from(self.out_edges(node))
        self.delete_node(node)


class DotStyle(dict):
    """
    Provides "styles" arithmetic for dot format
    See ./tests/test_dotsyle.py for examples
    """
    def __init__(self, val=None, **kw):
        """
        Dot styles as dict
        """
        val = self.from_str(val) if isinstance(val, str) else {} if val is None else val
        assert isinstance(val, dict)
        val.update(kw)
        super().__init__(val)

    @staticmethod
    def from_str(val: str):
        import re

        def parse_val(s, sv):
            if sv:
                for op in (int, float, str):
                    try:
                        s = op(sv)
                    except:
                        continue  # if failed - continue for the next casting attempt
                    break  # otherwise - exit with success
            return s  # last cast option -> str always succeeds

        found = re.findall(r'(\w+)\s*=\s*(?:"([^"]+)"|([^\s]+))', val)
        return {key: parse_val(s, sv) for key, s, sv in found}

    def to_str(self):
        return ' '.join([key + '=' + (f'"{v}"' if isinstance(v, str) and ' ' in v
                                      else f'{v}') for key, v in self.items()])

    def __iadd__(self, other):
        self.update(other)
        return self

    def __add__(self, other):
        res = self.__class__(self)
        res.update(other)
        return res

    def __radd__(self, other):
        return self.__class__(other) + self
