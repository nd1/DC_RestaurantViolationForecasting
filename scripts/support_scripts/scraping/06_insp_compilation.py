'''
Create a csv with inspection report, permit_id, date, time, inspec type, critical violation count, non-crit violation count, crit violation corrected on site, non-crit violation corrected on site, crit violation to be resolved, non-crit violation to be resolved, critical violation repeat violation, and non-crit violation repeat violation.

Create a second csv with inspection report, permit_id, critical violation count, non-crit violation count, lat, lon for mapping purposes.

20160623 Nicole Donnelly
'''

import pandas as pd

def map_list():
    #business list
    bus = pd.read_csv('geo_master.csv')
    #restaurant_inspections
    insp = pd.read_csv('restaurant_inspections.csv')

    merged =pd.merge(left=insp, right=bus, left_on='permit_id', right_on='permit_id')

    merged.drop(['name', 'address', 'category', 'permit_url',
       'map_url', 'phone_num', 'YelpID'], axis=1, inplace=True)

    merged.to_csv('mapping.csv', index=False)

def insp_compilation():

    #report  details list
    rpt_list = pd.read_csv('cleaned_report_results.csv')
    #inpsection report list
    insp_list = pd.read_csv('inpection_list_restauranttotal.csv')
    #violation list
    viol = pd.read_csv('cleaned_violation_list.csv')

    #select the inspection categories we are interested in
    insp_list = insp_list.loc[insp_list.insp_type.isin(['Routine','HACCP'])]
    #merge data from inps_list and viol and drop the report column
    merged = pd.merge(left=viol, right=insp_list, left_on='report', right_on='inspec_id')
    merged.drop('report', axis=1, inplace=True)

    #add the rpt_list data and drop extra columns
    merged = pd.merge(left=merged, right=rpt_list, left_on='inspec_id', right_on='report_id')
    merged.drop(['insp_url', 'report_id', 'risk', 'phone_num', 'insp_date_y', 'insp_timeout', 'insp_name', 'insp_badge'], axis=1, inplace=True)

    #reorder the columns
    new_order=['inspec_id', 'permit_id', 'insp_date_x', 'insp_timein', 'insp_type', 'violation', 'corrected_on_site', 'repeat_violation']
    df_new = merged[new_order]

    #create columns with the relevant info so we can create 1 row for each report
    for idx in df_new.index:
        if df_new.loc[idx, 'violation'] == 0:
            df_new.set_value(idx, 'crit_viol', 0)
            df_new.set_value(idx, 'non_crit_viol', 0)
            df_new.set_value(idx, 'crit_viol_cos', 0)
            df_new.set_value(idx, 'non_crit_viol_cos', 0)
            df_new.set_value(idx, 'crit_viol_rpt', 0)
            df_new.set_value(idx, 'non_crit_viol_rpt', 0)
        elif df_new.loc[idx, 'violation'] <= 40 and df_new.loc[idx, 'violation']>=1:
            df_new.set_value(idx, 'crit_viol', 1)
            df_new.set_value(idx, 'non_crit_viol', 0)
            if df_new.loc[idx, 'corrected_on_site'] == 1:
                df_new.set_value(idx, 'crit_viol_cos', 1)
                df_new.set_value(idx, 'non_crit_viol_cos', 0)
            else:
                df_new.set_value(idx, 'crit_viol_cos', 0)
                df_new.set_value(idx, 'non_crit_viol_cos', 0)
            if df_new.loc[idx, 'repeat_violation'] == 1:
                df_new.set_value(idx, 'crit_viol_rpt', 1)
                df_new.set_value(idx, 'non_crit_viol_rpt', 0)
            else:
                df_new.set_value(idx, 'crit_viol_rpt', 0)
                df_new.set_value(idx, 'non_crit_viol_rpt', 0)
        else:
            df_new.set_value(idx, 'non_crit_viol', 1)
            df_new.set_value(idx, 'crit_viol', 0)
            if df_new.loc[idx, 'corrected_on_site'] == 1:
                df_new.set_value(idx, 'non_crit_viol_cos', 1)
                df_new.set_value(idx, 'crit_viol_cos', 0)
            else:
                df_new.set_value(idx, 'non_crit_viol_cos', 0)
                df_new.set_value(idx, 'crit_viol_cos', 0)
            if df_new.loc[idx, 'repeat_violation'] == 1:
                df_new.set_value(idx, 'non_crit_viol_rpt', 1)
                df_new.set_value(idx, 'crit_viol_rpt', 0)
            else:
                df_new.set_value(idx, 'non_crit_viol_rpt', 0)
                df_new.set_value(idx, 'crit_viol_rpt', 0)
    df_new.to_csv('df_new.csv', index=False)
    df_new.drop(['violation', 'corrected_on_site', 'repeat_violation'], axis=1, inplace=True)

    df_new = df_new.groupby(['inspec_id', 'permit_id', 'insp_date_x', 'insp_timein', 'insp_type'])['crit_viol', 'non_crit_viol', 'crit_viol_cos', 'crit_viol_rpt', 'non_crit_viol_cos', 'non_crit_viol_rpt'].sum()
    df_new.reset_index(inplace=True)

    df_new['crit_viol_tbr'] = (df_new.crit_viol - df_new.crit_viol_cos)
    df_new['non_crit_viol_tbr'] = (df_new.non_crit_viol - df_new.non_crit_viol_cos)

    df_new.to_csv('restaurant_inspections.csv', index=False)

if __name__ == "__main__":
    insp_compilation()
    map_list()
