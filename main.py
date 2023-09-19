import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from model_fit import ZomatoModel
from model_fit import *
import pickle
st.set_page_config(
    page_title="Zomato Rating",
    layout="wide",
    page_icon=":food:",
    menu_items={
        'About': "Created By Rohit Tambavekar 'https://www.linkedin.com/in/rohit-tambavekar/'"
    }
)



hide_st_style = """
<style>
header { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

OnlineOrders = ["Yes", "No"]
TableBooking = ["Yes", "No"]
Location = ['BTM', 'HSR', 'Koramangala 5th Block', 'JP Nagar', 'Whitefield',
       'Indiranagar', 'Jayanagar', 'Marathahalli', 'Bannerghatta Road',
       'Electronic City', 'Bellandur', 'Koramangala 1st Block', 'Brigade Road',
       'Koramangala 7th Block', 'Koramangala 6th Block', 'Sarjapur Road',
       'Ulsoor', 'Koramangala 4th Block', 'Banashankari', 'MG Road',
       'Kalyan Nagar', 'Richmond Road', 'Frazer Town', 'Malleshwaram',
       'Basavanagudi', 'Residency Road', 'New BEL Road', 'Brookefield',
       'Kammanahalli', 'Banaswadi', 'Rajajinagar', 'Church Street',
       'Lavelle Road', 'Shanti Nagar', 'Shivajinagar', 'Cunningham Road',
       'Domlur', 'Old Airport Road', 'Ejipura', 'Commercial Street',
       'St. Marks Road', 'Koramangala 8th Block', 'Vasanth Nagar',
       'Jeevan Bhima Nagar', 'Wilson Garden', 'Bommanahalli',
       'Koramangala 3rd Block', 'Kumaraswamy Layout', 'Thippasandra',
       'Basaveshwara Nagar', 'Nagawara', 'Seshadripuram', 'Hennur',
       'HBR Layout', 'Infantry Road', 'Majestic', 'Race Course Road',
       'Yeshwantpur', 'City Market', 'ITPL Main Road, Whitefield',
       'Koramangala 2nd Block', 'South Bangalore',
       'Varthur Main Road, Whitefield', 'Kaggadasapura', 'Hosur Road',
       'CV Raman Nagar', 'Sanjay Nagar', 'RT Nagar', 'Vijay Nagar',
       'Sadashiv Nagar', 'Sahakara Nagar', 'Koramangala', 'Jalahalli',
       'Magadi Road', 'East Bangalore', 'Rammurthy Nagar', 'Sankey Road',
       'Langford Town', 'Old Madras Road', 'Mysore Road', 'Uttarahalli',
       'KR Puram', 'Kanakapura Road', 'Hebbal', 'North Bangalore',
       'Nagarbhavi', 'Kengeri', 'Central Bangalore', 'West Bangalore',
       'Jakkur', 'Yelahanka', 'Peenya', 'Rajarajeshwari Nagar']
RestaurantType = ['Casual Dining', 'Quick Bites', 'Cafe', 'Delivery',
       'Dessert Parlor', 'Bakery', 'Takeaway','Delivery',
       'Casual Dining', 'Bar', 'others']
Cuisines = ['North Indian, Mughlai, Chinese',
       'South Indian, North Indian', 'North Indian', 'Cafe',
       'Cafe, Continental', 'Cafe, Fast Food', 'Cafe, Bakery',
       'Bakery, Desserts', 'Pizza', 'Biryani',
       'North Indian, Chinese, Fast Food', 'Chinese, Thai, Momos',
       'South Indian', 'Burger, Fast Food', 'Pizza, Fast Food',
       'North Indian, Chinese', 'Chinese, Thai', 'Ice Cream, Desserts',
       'Biryani, Fast Food', 'Fast Food, Burger', 'Desserts, Beverages',
       'Chinese', 'Bakery', 'Biryani, South Indian', 'Fast Food',
       'South Indian, Chinese, North Indian', 'Mithai, Street Food',
       'South Indian, Chinese', 'Biryani, North Indian, Chinese',
       'Desserts', 'Ice Cream', 'South Indian, North Indian, Chinese',
       'South Indian, Biryani', 'Beverages', 'Mithai',
       'North Indian, Street Food', 'Chinese, North Indian',
       'South Indian, North Indian, Chinese, Street Food', 'Andhra',
       'Italian, Pizza', 'Street Food', 'Arabian',
       'North Indian, Chinese, Continental', 'Desserts, Ice Cream',
       'North Indian, Chinese, Biryani', 'Fast Food, Rolls',
       'Beverages, Fast Food', 'North Indian, Chinese, South Indian',
       'South Indian, Fast Food', 'North Indian, Fast Food',
       'Beverages, Desserts', 'North Indian, Continental',
       'North Indian, South Indian', 'North Indian, Biryani',
       'Finger Food', 'Continental', 'Fast Food, Beverages',
       'Andhra, Biryani', 'Biryani, Kebab', 'North Indian, Mughlai',
       'North Indian, South Indian, Chinese', 'Cafe, Desserts',
       'Biryani, North Indian', 'Chinese, Momos', 'Kerala, South Indian',
       'Desserts, Bakery', 'Bakery, Fast Food', 'Kerala',
       'North Indian, Chinese, Seafood', 'others']
Type = ['Delivery', 'Drinks & nightlife', 'Dine-out', 'Cafes', 'Desserts',
       'Buffet', 'Pubs and bars']


def main():
    # Create an instance of the ZomatoModel class
    zomato_model = ZomatoModel()
    train = pd.read_csv('zomato_train.csv')
    test = pd.read_csv('zomato_test.csv')
    # Load the trained model from the pickled file
    zomato_model.unpickle_model()
    zomato_model.unpickle_encodings()
    zomato_model.unpickle_ann_model()
    zomato_model.unpickle_sc_model()
    left_tit,center_tit ,right_tit = st.columns([15,10,15])
    with center_tit:
        st.title(":red[Zomato Ratings]")
    selected = option_menu(
        menu_title=None,
        options = ["Home","Dashboard","Contact"],
        icons=["house","kaban","envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal"
    )
    
    if selected == "Home":
        col1, col2,col3 = st.columns([5,5,5])
        
        with col1:
            online_order = st.selectbox("Online Orders:",OnlineOrders)
            table_booking = st.selectbox("Table Booking :",TableBooking)
            location = st.selectbox("Location :",Location)
            restaurant_type = st.selectbox("Restaurant Type :",RestaurantType)
            
                 
            
        with col2:
            cuisines = st.selectbox(" Cuisines :",Cuisines)
            type = st.selectbox(" Restauranted listed in Type :",Type)
            cost_for_two = st.number_input('Cost for two :',min_value = 60,max_value = 6000,value = 500,step = 10)
            Votes = st.number_input('Votes :',min_value = 0,max_value = 6000,value = 100,step = 1)
        
        with col3:    
            st.write("Check ratings for the Restaurant")
            if st.button('Check Ratings'):
                zomato_model.compute_location_freq_map(train)
                ol_order,tb_book,loc_hotel,cui_hotel= zomato_model.input_encodings(online_order,table_booking,location,cuisines)
                type_hotel = zomato_model.label_encoder.transform([type])[0]
                # Votes,loc_hotel,cost_for_two,cui_hotel,ol_order,tb_book,type_hotel
                data = {
                        'votes': [Votes],
                        'location': [loc_hotel],
                        'cost_for_two': [cost_for_two],
                        'cuisines': [cui_hotel],
                        'online_order': [ol_order],
                        'book_table': [tb_book],
                        'type': [type_hotel]
                    }
                single_row_df = pd.DataFrame(data)

                # Make predictions using the trained model
                y_single_row_pred = zomato_model.rft.predict(single_row_df)[0]
                st.write(f"The Random forest rating for the Restaurant is :{round(y_single_row_pred,1)}")     
            
            if st.button('Check ANN Ratings'):
                zomato_model.compute_location_freq_map(train)
                ol_order,tb_book,loc_hotel,cui_hotel= zomato_model.input_encodings(online_order,table_booking,location,cuisines)
                type_hotel = zomato_model.label_encoder.transform([type])[0]
                # Votes,loc_hotel,cost_for_two,cui_hotel,ol_order,tb_book,type_hotel
                data_ann = {
                        'votes': [Votes],
                        'location': [loc_hotel],
                        'cost_for_two': [cost_for_two],
                        'cuisines': [cui_hotel],
                        'online_order': [ol_order],
                        'book_table': [tb_book],
                        'type': [type_hotel]
                    }
                
                # List of restaurant types
                

                # Create a list of keys with '_rest_type' suffix
                restaurant_types_list = [f'{r_type}'+'_rest_type' for r_type in RestaurantType]
                
                # Append keys with values initialized to 0 to the data dictionary
                for r_type in restaurant_types_list:
                    data_ann[r_type] = [0]
                data_ann[restaurant_type.replace(" ", "_")+'_rest_type'] = [1]
                single_row_df = pd.DataFrame(data_ann)
                desired_columns = [
                                'online_order',
                                'book_table',
                                'votes',
                                'location',
                                'cuisines',
                                'cost_for_two',
                                'type',
                                'Bakery_rest_type',
                                'Bar_rest_type',
                                'Cafe_rest_type',
                                'Casual Dining_rest_type',
                                'Delivery_rest_type',
                                'Dessert Parlor_rest_type',
                                'Quick Bites_rest_type',
                                'Takeaway_rest_type',
                                'others_rest_type'
                            ]

                # Reorder the columns
                single_row_df = single_row_df[desired_columns]

                # Sort the columns according to the desired order
                input_data = zomato_model.sc.transform(single_row_df)
                # Then, make predictions using the ANN model
                predictions = zomato_model.ann.predict(input_data)[0]
                preds = np.round(predictions,1)
                # Print the predictions
                st.write(f'Ann Ratings for the Restaurant is : {np.round(preds,1)}')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()