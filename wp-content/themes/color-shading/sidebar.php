			<div id="sidebar">
              <?php if ( !function_exists('dynamic_sidebar') || !dynamic_sidebar() ) : ?>
              <div class="widget">
                <h3>Categories</h3>
                <ul id="categories">
			      <?php wp_list_categories('title_li=&number=7'); ?>
                </ul><!-- /categories -->
              </div><!-- /widget -->
              <div class="widget">
                <h3>Blogroll</h3>
                <ul id="blogroll">
			      <?php wp_list_bookmarks('categorize=0&title_li='); ?>
                </ul><!-- /blogroll -->
              </div><!-- /widget -->
              <div class="widget">
                <h3>Recent Comments</h3>
                <ul id="widgetacc">
                  <?php latest_comments(); ?>
                </ul>
              </div><!-- /widget -->
              <?php endif; ?>
              <div class="clear"></div>
              <?php wp_meta(); ?>
            </div><!-- /sidebar -->