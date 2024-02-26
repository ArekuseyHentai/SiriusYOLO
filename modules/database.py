import sqlite3
import datetime
import modules.funcs
rec = modules.funcs.Rec()


class Times():
    '''
    Times class
    '''
    def __init__(self) -> None:
        self.connection = sqlite3.connect('./src/test_sql.db')
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Times (
                datetime TEXT,
                number INTEGER
                )
                ''')
        self.connection.commit()

    class TimesResponse():
        "TimesResponse class"
        def __init__(self, result: any) -> None:
            self.datatime = int(result[0])
            self.number = int(result[1])

    def create(self):
        now = datetime.datetime.now()
        value = str(now.isoweekday())+","+str(now.hour)
        num = rec.get_persons()
        self.cursor.execute('''
                        INSERT INTO Times (datetime,number) VALUES (?,?)
                            ''', (value, num))
        self.connection.commit()

    def many(self, hour: str, weekday: str) -> list[TimesResponse] | None:
        value = weekday+","+str
        self.cursor.execute(
                'SELECT * FROM Guilds WHERE datetime = ?',
                (value,)
                )
        result = self.cursor.fetchall()
        if len(result) == 0:
            return None
        res = []
        for item in result:
            res.append(self.TimesResponse(item))
        return res
