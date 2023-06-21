import duckdb
import pandas
from soda.scan import Scan

input_df = pandas.DataFrame(dict(destination=["New York", "New York", "Chicago"]))

with duckdb.connect(":memory:") as con:
    con.sql(
        """\
        CREATE VIEW aggregated_destinations AS
        SELECT destination, COUNT(destination) as cnt
        FROM input_df
        GROUP BY destination
        """
    )
    scan = Scan()
    scan.add_duckdb_connection(con)
    scan.set_data_source_name("duckdb")
    scan.add_sodacl_yaml_files("soda-sql/checks.yml")
    scan.set_scan_definition_name("test_destinations")
    scan.execute()
    scan.assert_no_checks_fail()
    print(scan.get_logs_text())

try:
    with duckdb.connect(":memory:") as con:
        con.sql(
            """\
        CREATE VIEW aggregated_destinations AS
        SELECT destination, COUNT(destination) as counted_destinations
        FROM input_df
        GROUP BY destination
        """
        )
        scan = Scan()
        scan.add_duckdb_connection(con)
        scan.set_data_source_name("duckdb")
        scan.add_sodacl_yaml_files("soda-sql/checks.yml")
        scan.set_scan_definition_name("test_destinations")
        scan.execute()
        scan.assert_no_checks_fail()
        print(scan.get_logs_text())
except AssertionError as e:
    print(e)

with duckdb.connect("database.db") as con:
    con.sql(
        """\
        CREATE VIEW aggregated_destinations AS
        SELECT destination, COUNT(destination) as cnt
        FROM input_df
        GROUP BY destination
        """
    )
    scan = Scan()
    scan.add_duckdb_connection(con)
    scan.set_data_source_name("my_data")
    scan.add_configuration_yaml_file("soda-sql/configuration.yml")
    scan.add_sodacl_yaml_files("soda-sql/checks.yml")
    scan.set_scan_definition_name("test_destinations")
    scan.execute()
    scan.assert_no_checks_fail()
    print(scan.get_logs_text())
