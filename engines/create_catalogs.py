from algutils.logs import setup_logs
import click
from toolbox.engines import engines


setup_logs(name_from=__file__, debug='engines')


@click.command()
@click.option('-o', '--overwrite', is_flag=True,
              help='Allow to overwrite found catalogs')
@click.option('-p', '--path', multiple=True,
              help='Path to folder or module of a catalog')
@click.option('-m', '--module', multiple=True,
              help='Package or module (dotted) pointing to the location of catalog')
@click.option('-d', '--deep', default=0, type=int,
              help='Levels to search under the specified location')
@click.option('--dry', default=False, help='Dry run only print intentions')
def create_catalogs(overwrite, path, module, deep, dry):
    print(f"Creating engine catalogs with options: {overwrite=}, {deep=}, {dry=}")
    print(f"{path=}\n{module=}")

    if not path and not module:
        catalogs = engines.discover_catalogs()
        print(f'Found {len(catalogs)} in environment based paths')
        path = [cat.name for cat in catalogs]

    for p in path:
        print(f'Creating for path {p}: {overwrite=}, {deep=}')
        if not dry:
            n = engines.create_catalog(path=p, overwrite=overwrite, deep=deep)
            print(f'Added {n} engines')

    for m in module:
        print(f'Creating for module {m}: {overwrite=}, {deep=}')
        if not dry:
            n = engines.create_catalog(package=m, overwrite=overwrite, deep=deep)
            print(f'Added {n} engines')


if __name__ == '__main__':
    create_catalogs()
