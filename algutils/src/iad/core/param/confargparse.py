import configargparse as cap


class ArgumentParserYAML(cap.ArgumentParser):
    def __init__(self, *, hash_groups=None, add_env_var_help=False, **kws):
        super().__init__(config_file_parser_class=cap.YAMLConfigFileParser,
                         add_env_var_help=add_env_var_help,
                         **kws)
        self.hash_groups = hash_groups or {}

    def save_config(self, namespace, config_file):
        if BUG_FIXED := False:  # TODO: write_config_file prints all values as str!
            self.write_config_file(namespace, [str(config_file)])
        else:
            op_map = {op.dest: op.option_strings[-1] for op in self._get_optional_actions()}

            def items_to_save():
                for k, v in namespace.__dict__.items():
                    if not (v is None or k in ('cmd', 'config')):
                        yield op_map[k].strip('-'), v

            file_contents = self._config_file_parser.serialize(items_to_save())
            with self._config_file_open_func(config_file, "w") as output_file:
                output_file.write(file_contents)
                print(f"Config file saved: {config_file}")

    def hashes(self, ns: cap.Namespace, length=4) -> dict:
        """
        Calculate hash values for the defined groups of arguments.
        Return dict with hashes py group.
        If groups are not defined the namespace is used and named 'cfg'.

        :param ns: Namespace with parsed arguments
        :param length: length of hash string tail to used (0 - all of it)

        :return: dict {grp_name: hash_str}
        """
        from .tbox import TBox
        groups = self.hash_groups or {'cfg': [*vars(ns)]}
        ns = vars(ns)

        def hash_group(names):
            group_params = {name: ns[name] for name in names if name in ns}
            return TBox(group_params).hash_str(length)

        return {grp: hash_group(names) for grp, names in groups.items()}

    def hash_stamp(self, ns: cap.Namespace, length=4) -> str:
        """
        Hash stamp string unique for the ns content by arguments groups.
        :param ns:  namespace with arguments
        :param length: length of hash strings
        :return: "grp1_hash1_grp2_hash2_..."
        """
        return '_'.join(map(lambda it: f"{it[0]}_{it[1]}", self.hashes(ns, length).items()))
